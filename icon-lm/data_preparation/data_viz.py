import torch
import data_io as io
from absl import app, flags, logging
import random
from pprint import pprint
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
import pickle

# print torch visiable devices
print('torch.cuda.is_available()', torch.cuda.is_available())

def remove_duplicates(lst):
    # remove duplicates while preserving order
    seen = set()
    return [x for x in lst if x not in seen and not seen.add(x)]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_embedding(string_list, tokenizer, model, model_name):
  inputs = tokenizer(string_list, return_tensors="pt", padding=True).to(device)
  print([(a, inputs[a].shape) for a in inputs])
  if model_name == 'gpt2':
    mask = inputs['attention_mask'].detach().cpu().numpy().astype(bool) # (batch_size, seq_len)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.detach().cpu().numpy() # (batch_size, seq_len, hidden_size)
    embeddings = np.mean(embeddings, axis=1, keepdims=False, where= mask[:, :, None])
  elif model_name == 'gpt2-icon':
    mask = inputs['attention_mask'].detach().cpu().numpy().astype(bool) # (batch_size, seq_len)
    outputs = model(inputs['input_ids'], attention_mask=None) # use default causal mask
    embeddings = outputs[0].detach().cpu().numpy()  # (batch_size, seq_len, hidden_size)
    embeddings = np.mean(embeddings, axis=1, keepdims=False, where= mask[:, :, None])
  elif model_name == 'bert-base-uncased':
    embeddings = outputs.pooler_output.detach().cpu().numpy()
  return embeddings

def get_embedding_dict(caption_dir, tokenizer, model, model_name):
  
  problem_list = ['ode1', 'ode2', 'ode3', 'pde1', 'pde2', 'pde3', 'series', 'mfc_gparam', 'mfc_rhoparam']
  embedding_dict = {}
  for problem in problem_list:
    l1,l2 = io.read_lists_from_file('{}/{}.md'.format(caption_dir, problem), mode='separate')
    embedding_dict[(problem, 'g1')] = get_embedding(l1, tokenizer, model, model_name)
    embedding_dict[(problem, 'g2')] = get_embedding(l2, tokenizer, model, model_name)
    print(problem, embedding_dict[(problem, 'g1')].shape, embedding_dict[(problem, 'g2')].shape)

  return embedding_dict

def plot_embedding(caption_dir, embedding_dict, problem_list, title, num = 100): 
  # plot embedding in the same figure
  
  label_dict = {'ode1': 'ODE 1', 'ode2': 'ODE 2', 'ode3': 'ODE 3', 
                'pde1': 'PDE 1', 'pde2': 'PDE 2', 'pde3': 'PDE 3', 
                'series': 'Damped Oscillator', 
                'mfc_gparam': 'MFC g-parameter', 'mfc_rhoparam': 'MFC rho-parameter'}
  color_list = sns.color_palette("husl", len(problem_list))

  plt.figure(figsize=(6,6))
  all_data = []
  all_label = []
  for (i,problem) in enumerate(problem_list):
    all_data.append(embedding_dict[(problem, 'g1')][:num])
    all_label.append(np.ones(num)*i)
    all_data.append(embedding_dict[(problem, 'g2')][:num])
    all_label.append(np.ones(num)*i)
  all_data = np.concatenate(all_data, axis=0)
  all_label = np.concatenate(all_label, axis=0)
  print("all_data, all_label", all_data.shape, all_label.shape)

  print("silhouette_score, full feature", silhouette_score(all_data, all_label))
  for (method, coms) in [(TSNE, 2), (PCA,100), (PCA,10), (Isomap,100), (Isomap,10)]:
      emd = method(n_components=coms).fit_transform(all_data)
      print("silhouette_score, {}-{}".format(str(method), coms), silhouette_score(emd, all_label))
  
  X_embedded = TSNE(n_components=2).fit_transform(all_data)
  for i, problem in enumerate(problem_list):
    X_embedded_g1 = X_embedded[i*2*num:i*2*num+num]
    X_embedded_g2 = X_embedded[i*2*num+num:(i+1)*2*num]
    plt.scatter(X_embedded_g1[:,0], X_embedded_g1[:,1], label=label_dict[problem], color=color_list[i], marker='o')
    plt.scatter(X_embedded_g2[:,0], X_embedded_g2[:,1], color=color_list[i], marker='^')

  # 2 columns
  width = np.max(X_embedded[:,1]) - np.min(X_embedded[:,1])
  plt.ylim(np.min(X_embedded[:,1])- width * 0.4, np.max(X_embedded[:,1])+ width * 0.05)
  plt.legend(ncol=2, loc = 'lower left')

  plt.title(title)
  plt.savefig('{}/{}_embedding.png'.format(caption_dir, title.replace(' ', '_') + "_" + "_".join(problem_list)))
  plt.close()

def get_trained_model():
  
  import sys
  sys.path.append('../')
  from runner_torch import Runner
  from dataloader import DataProvider, print_eqn_caption, split_data # import in function to enable flags in dataloader
  import utils
  import jax.tree_util as tree

  test_data_dirs = ['/home/shared/icon/data/data0910c']
  test_file_names = ["{}/{}".format(i, j) for i in test_data_dirs for j in ['test*']]

  
  test_config = utils.load_json("../config_data/" + "test_lm_precise_config.json")
  model_config = utils.load_json("../config_model/" + "model_gpt2_config.json")
  test_data = DataProvider(seed = 1,
                            config = test_config,
                            file_names = test_file_names,
                            batch_size = 8,
                            deterministic = False,
                            drop_remainder = False, 
                            shuffle_dataset = False,
                            num_epochs=1,
                            shuffle_buffer_size=10,
                            num_devices=0,
                            real_time = True,
                            caption_home_dir = '../data_preparation',
                          )
  
  equation, caption, data, label = test_data.get_next_data()
  print_eqn_caption(equation, caption, decode = False)
  print(tree.tree_map(lambda x: x.shape, data)) 

  opt_config = {'peak_lr': 0.001,
                'end_lr': 0,
                'warmup_steps': 10,
                'decay_steps': 100,
                'gnorm_clip': 1,
                'weight_decay': 0.0001,
                }
  
  runner = Runner(data, model_config, opt_config = opt_config, 
                    model_name = 'gpt2', pretrained = True, 
                    trainable_mode = 'all',
                    loss_mode = ['cap','nocap'],
                    )
  runner.restore(FLAGS.restore_dir, FLAGS.restore_step, restore_opt_state=False)
  model = runner.model.module if hasattr(runner.model, 'module') else runner.model
  model = model.gpt2
  return model

def main(argv):
  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)

  caption_dir_base = 'captions_1009'
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token

  print("-----------Fine-tuned GPT-2----------")
  try :
    embedding_dict_finetuned = pickle.load(open('{}/embedding_dict_finetuned.pkl'.format(caption_dir_base), 'rb'))
  except:
    trained_model = get_trained_model().to(device)
    print(trained_model)
    embedding_dict_finetuned = get_embedding_dict(caption_dir_base, tokenizer, trained_model, 'gpt2-icon')
    pickle.dump(embedding_dict_finetuned, open('{}/embedding_dict_finetuned.pkl'.format(caption_dir_base), 'wb'))
    del trained_model
  plot_embedding(caption_dir_base, embedding_dict_finetuned, FLAGS.problem, 'Fine-tuned GPT-2', num = 100)

  print("-----------Default GPT-2----------")
  try :
    embedding_dict_default = pickle.load(open('{}/embedding_dict_default.pkl'.format(caption_dir_base), 'rb'))
  except:
    default_model = AutoModel.from_pretrained('gpt2').to(device)
    embedding_dict_default = get_embedding_dict(caption_dir_base, tokenizer, default_model, 'gpt2')
    pickle.dump(embedding_dict_default, open('{}/embedding_dict_default.pkl'.format(caption_dir_base), 'wb'))
    del default_model
  plot_embedding(caption_dir_base, embedding_dict_default, FLAGS.problem, 'Standard GPT-2', num = 100)

  
if __name__ == "__main__":


  FLAGS = flags.FLAGS

  flags.DEFINE_list('problem', ['ode1', 'ode2', 'ode3', 'series', 'pde1', 'pde2', 'pde3', 'mfc_gparam', 'mfc_rhoparam'], 'problem to process')
  flags.DEFINE_string('restore_dir', '/home/shared/icon/save/yl/ckpts/icon_gpt2_full/20231014-194955', 'restore directory')
  flags.DEFINE_integer('restore_step', 1000000, 'restore step')

  app.run(main)