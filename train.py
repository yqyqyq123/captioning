import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from attentionModel import BahdanauAttnDecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.CenterCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    transform_val = transforms.Compose([ 
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)
    val_loader = get_loader(args.val_dir, args.val_caption_path, vocab,
                             transform_val, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    encoder.freeze_bottom()
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
#     decoder = BahdanauAttnDecoderRNN(args.hidden_size, args.embed_size, len(vocab)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    accs, b1s, b2s, b3s, b4s = [], [], [], [], []
    for epoch in range(args.num_epochs):
        decoder.train()
        encoder.train()
        losses = []
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch+1, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        
#         acc, b1, b2, b3, b4 = evaluate(val_loader, encoder, decoder, vocab)
#         accs.append(acc)
#         b1s.append(b1)
#         b2s.append(b2)
#         b3s.append(b3)
#         b4s.append(b4)
        avg_loss = sum(losses)/total_step
        
        print('Epoch {} Average Training Loss: {:.4f}'.format(epoch+1, avg_loss))
        
        with open ('stem_freeze_freq1000.txt', 'a') as file:
            file.write("Epoch {} \n".format(epoch+1))
            file.write('Average Accuracy: {} \n'.format(acc))
            file.write('Average Loss: {} \n'.format(avg_loss))
            file.write('Average BLEU gram1: {} \n'.format(b1))
            file.write('Average BLEU gram2: {} \n'.format(b2))
            file.write('Average BLEU gram3: {} \n'.format(b3))
            file.write('Average BLEU gram4: {} \n'.format(b4))
            file.write('\n')
                
    plt.title("Accuracy vs BLEU score")
    plt.plot(np.arange(1,args.num_epochs+1),accs, label='accuracy')
    plt.plot(np.arange(1,args.num_epochs+1),b1s, label='BLEU 1')
    plt.plot(np.arange(1,args.num_epochs+1),b2s, label='BLEU 2')
    plt.plot(np.arange(1,args.num_epochs+1),b3s, label='BLEU 3')
    plt.plot(np.arange(1,args.num_epochs+1),b4s, label='BLEU 4')
    plt.xlabel("epochs")
    plt.xticks(np.arange(1,args.num_epochs+1))
    plt.legend(loc='upper left')
    plt.savefig('accuracy_BLEU.png')
    plt.clf()


def evaluate(data_loader, encoder, decoder, vocab):
    encoder.eval()
    decoder.eval()
    total_step = len(data_loader)
    num = 0
    accuracy = 0
    bleu_gram1 = 0
    bleu_gram2= 0
    bleu_gram3 = 0
    bleu_gram4 = 0
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(data_loader):

                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                topv, topi = outputs.topk(1, dim=1)
                targets = targets.unsqueeze(-1)
#                 print("TARGETS", targets)
#                 print("TOPI", topi)
                accuracy += float((topi == targets).sum())/targets.shape[0]
                sentence_length=0
                for j in range(len(lengths)):
                    candidate = [vocab.idx2word[int(idx[0])] for idx in topi[sentence_length:sentence_length+lengths[j]]]
                    reference = [[vocab.idx2word[int(idx[0])] for idx in targets[sentence_length:sentence_length+lengths[j]]]]
                    bleu_gram1 += float(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))/len(lengths)
                    bleu_gram2 += sentence_bleu(reference,candidate,weights=(0.5,0.5,0,0))/len(lengths)
                    bleu_gram3 += sentence_bleu(reference,candidate,weights=(0.33,0.33,0.33,0))/len(lengths)
                    bleu_gram4 += sentence_bleu(reference,candidate,weights=(0.25,0.25,0.25,0.25))/len(lengths)
                    sentence_length+=lengths[j]
        print('Average Accuracy: {}'.format(accuracy/total_step))
        print('Average BLEU gram1: {}'.format(bleu_gram1/total_step))
        print('Average BLEU gram2: {}'.format(bleu_gram2/total_step))
        print('Average BLEU gram3: {}'.format(bleu_gram3/total_step))
        print('Average BLEU gram4: {}'.format(bleu_gram4/total_step))

        
        return accuracy/total_step, bleu_gram1/total_step, bleu_gram2/total_step, bleu_gram3/total_step, bleu_gram4/total_step
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./vocab_lemma.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./resized2014/', help='directory for resized training images')
    parser.add_argument('--val_dir', type=str, default='./resized2014_val', help='directory for resized validation images')
    parser.add_argument('--caption_path', type=str, default='./datasets/coco2014/trainval_coco2014_captions/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='./datasets/coco2014/trainval_coco2014_captions/captions_val2014.json', help='path for validation annotation json file')                    
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)