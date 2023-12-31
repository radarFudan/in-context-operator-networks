import torch
import data_io as io
from absl import app, flags, logging
import random
import os

# print torch visiable devices
print('torch.cuda.is_available()', torch.cuda.is_available())

def remove_duplicates(lst):
    # remove duplicates while preserving order
    seen = set()
    return [x for x in lst if x not in seen and not seen.add(x)]


def read_and_merge(caption_dirs, target_dir, problem):
  random.seed(0)

  L1 = []
  L2 = []
  for caption_dir in caption_dirs:
    try:
      l1,l2 = io.read_lists_from_file('{}/{}.md'.format(caption_dir, problem), mode='separate')
      L1.extend(l1)
      L2.extend(l2)
    except:
      print('problem {} not found in {}'.format(problem, caption_dir))
      pass
  # remove duplicates
  L1 = remove_duplicates(L1)
  L2 = remove_duplicates(L2)
  # shuffle list
  random.shuffle(L1)
  random.shuffle(L2)
  
  # select 100
  print(problem, len(L1), len(L2))
  assert len(L1) >= 100
  assert len(L2) >= 100
  L1 = L1[:100]
  L2 = L2[:100]

  # add [] to 0.001, 0.002, 0.003, 0.004 to avoid error during training, say 0.001 be replaced with 0.002 and replaced again
  for i in range(len(L1)):
    L1[i] = L1[i].replace("0.001", "[0.001]").replace("0.002", "[0.002]").replace("0.003", "[0.003]").replace("0.004", "[0.004]")
  for i in range(len(L2)):
    L2[i] = L2[i].replace("0.001", "[0.001]").replace("0.002", "[0.002]").replace("0.003", "[0.003]").replace("0.004", "[0.004]")
  # write two list into one file, with a blank line between them
  try:
    os.mkdir(target_dir)
  except:
    pass
  io.write_whole_file('{}/{}.md'.format(target_dir, problem), '\n'.join(L1) + '\n\n' + '\n'.join(L2))


def main(argv):


  for key, value in FLAGS.__flags.items():
    print(value.name, ": ", value._value, flush=True)


  caption_dir_base = 'captions_1009'
  caption_dirs = [caption_dir_base+ c for c in ['a','b','c','d','e','f','g','h']]

  for problem in FLAGS.problem:
    read_and_merge(caption_dirs, caption_dir_base, problem)
  
  
if __name__ == "__main__":


  FLAGS = flags.FLAGS

  flags.DEFINE_list('problem', ['ode1', 'ode2', 'ode3', 'pde1', 'pde2', 'pde3', 'series', 'mfc_gparam', 'mfc_rhoparam'], 'problem to process')

  app.run(main)