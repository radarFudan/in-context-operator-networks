import numpy as np
import jax.numpy as jnp
import jax
from einshape import jax_einshape as einshape
import pickle
from functools import partial

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
sys.path.append('../')
sys.path.append('weno/')
import utils
from absl import app, flags, logging
import haiku as hk
import matplotlib.pyplot as plt

import data_writetfrecord as datawrite
import data_utils
from weno.weno_solver import generate_weno_scalar_sol, generate_weno_euler_sol


def generate_conservation_weno_burgers(seed, eqns, quests, length, steps, dt, num, name):
  '''du/dt = a * c + b'''
  eqn_type = "conservation_weno_burgers"
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  coeffs_burgers = np.linspace(0.0, 1, eqns+1)[1:] # the coefficient of the nonlinear term, 0.5 is standard burgers
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, coeff_burgers in enumerate(coeffs_burgers):
    fn = jax.jit(lambda u: coeff_burgers * u * u)
    for j in range(quests):
      init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N+1)
      sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps) # (num, steps + 1, N, 1)
      all_xs.append(xs) # (N,)
      all_us.append(sol) # (num, steps + 1, N, 1)
      all_params.append("{:.3f}".format(coeff_burgers))
      all_eqn_captions.append(['dummy caption'])
    utils.print_dot(i)
  for ptype in ['forward']:
    write_evolution_tfrecord(seed = next(rng)[0], eqn_type = eqn_type, 
                      all_params = all_params, all_eqn_captions = all_eqn_captions,
                      all_xs = all_xs, all_us = all_us, stride = 100,
                      problem_type = ptype, file_name = "{}_{}_{}.tfrecord".format(name, eqn_type, ptype))


def generate_conservation_weno_quadratic(seed, eqns, quests, length, steps, dt, num, name, eqn_mode):
  '''du/dt + d(a * u^2 + b * u)/dx = 0'''
  eqn_type = "conservation_weno_quadratic"
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if 'random' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    coeffs_a = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_b = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
  elif 'grid' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    # meshgrid of coeffs_a and coeffs_b
    coeffs_a = np.linspace(minval, maxval, eqns) # the coefficient of the quadratic term
    coeffs_b = np.linspace(minval, maxval, eqns) # the coefficient of the linear term
    coeffs_a, coeffs_b = np.meshgrid(coeffs_a, coeffs_b)
    coeffs_a = coeffs_a.flatten()
    coeffs_b = coeffs_b.flatten()
  else:
    raise NotImplementedError("eqn_mode = {} is not implemented".format(FLAGS.eqn_mode))
  for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
    print("coeff_a = {:.3f}, coeff_b = {:.3f}".format(coeff_a, coeff_b), flush=True)
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b) in enumerate(zip(coeffs_a, coeffs_b)):
    fn = jax.jit(lambda u: coeff_a * u * u + coeff_b * u)
    grad_fn = jax.jit(lambda u: 2 * coeff_a * u + coeff_b)
    for j in range(quests):
      init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N+1)
      sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn) # (num, steps + 1, N, 1)
      assert not jnp.any(jnp.isnan(sol))
      all_xs.append(xs) # (N,)
      all_us.append(sol) # (num, steps + 1, N, 1)
      all_params.append("{:.8f}_{:.8f}".format(coeff_a, coeff_b))
      all_eqn_captions.append(['dummy caption'])
    utils.print_dot(i)
    if (i+1) % (len(coeffs_a)//FLAGS.file_split) == 0 or i == len(coeffs_a) - 1:
      for ptype in ['forward', 'backward']:
        write_evolution_tfrecord(seed = next(rng)[0], eqn_type = eqn_type, 
                          all_params = all_params, all_eqn_captions = all_eqn_captions,
                          all_xs = all_xs, all_us = all_us, stride = 100,
                          problem_type = ptype, file_name = "{}_{}_{}_{}.tfrecord".format(name, eqn_type, ptype, i+1))
      all_xs = []; all_us = []; all_params = []; all_eqn_captions = []



def generate_conservation_weno_cubic(seed, eqns, quests, length, steps, dt, num, name, eqn_mode):
  '''du/dt + d(a * u^2 + b * u)/dx = 0'''
  eqn_type = "conservation_weno_cubic"
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if 'random' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    coeffs_a = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_b = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_c = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
  elif 'grid' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    values = np.linspace(minval, maxval, eqns)
    coeffs_a, coeffs_b, coeffs_c = np.meshgrid(values, values, values)
    coeffs_a = coeffs_a.flatten()
    coeffs_b = coeffs_b.flatten()
    coeffs_c = coeffs_c.flatten()
  else:
    raise NotImplementedError("eqn_mode = {} is not implemented".format(FLAGS.eqn_mode))
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    print("coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(coeff_a, coeff_b, coeff_c), flush=True)
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    fn = jax.jit(lambda u: coeff_a * u * u * u + coeff_b * u * u + coeff_c * u)
    grad_fn = jax.jit(lambda u: 3 * coeff_a * u * u + 2 * coeff_b * u + coeff_c)
    for j in range(quests):
      while True:
        init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N+1)
        if jnp.max(jnp.abs(init)) < 3.0:
          break
      sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, stable_tol = 10.0) # (num, steps + 1, N, 1)
      all_xs.append(xs) # (N,)
      all_us.append(sol) # (num, steps + 1, N, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a, coeff_b, coeff_c))
      all_eqn_captions.append(['dummy caption'])
    utils.print_dot(i)
    if (i+1) % (len(coeffs_a)//FLAGS.file_split) == 0 or i == len(coeffs_a) - 1:
      for ptype in ['forward', 'backward']:
        for st in FLAGS.stride:
          sti = int(st)
          write_evolution_tfrecord(seed = next(rng)[0], eqn_type = eqn_type, 
                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                            all_xs = all_xs, all_us = all_us, stride = sti,
                            problem_type = ptype, file_name = "{}_{}_{}_stride{}_{}.tfrecord".format(name, eqn_type, ptype, sti, i+1))
      all_xs = []; all_us = []; all_params = []; all_eqn_captions = []


def generate_conservation_weno_sin(seed, eqns, quests, length, steps, dt, num, name, eqn_mode):
  '''du/dt + d(a * u^2 + b * u)/dx = 0'''
  eqn_type = "conservation_weno_sin"
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  if 'random' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    coeffs_a = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_b = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
    coeffs_c = jax.random.uniform(next(rng), shape = (eqns,), minval = minval, maxval = maxval)
  elif 'grid' in eqn_mode:
    minval = float(FLAGS.eqn_mode.split('_')[1])
    maxval = float(FLAGS.eqn_mode.split('_')[2])
    values = np.linspace(minval, maxval, eqns)
    coeffs_a, coeffs_b, coeffs_c = np.meshgrid(values, values, values)
    coeffs_a = coeffs_a.flatten()
    coeffs_b = coeffs_b.flatten()
    coeffs_c = coeffs_c.flatten()
  elif 'fix' in eqn_mode:
    val_0 = float(FLAGS.eqn_mode.split('_')[1])
    val_1 = float(FLAGS.eqn_mode.split('_')[2])
    val_2 = float(FLAGS.eqn_mode.split('_')[3])
    coeffs_a = np.array([val_0])
    coeffs_b = np.array([val_1])
    coeffs_c = np.array([val_2])
  else:
    raise NotImplementedError("eqn_mode = {} is not implemented".format(FLAGS.eqn_mode))
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    print("coeff_a = {:.3f}, coeff_b = {:.3f}, coeff_c = {:.3f}".format(coeff_a, coeff_b, coeff_c), flush=True)
  xs = jnp.linspace(0.0, 1.0, length, endpoint=False)
  all_xs = []; all_us = []; all_params = []; all_eqn_captions = []
  for i, (coeff_a, coeff_b, coeff_c) in enumerate(zip(coeffs_a, coeffs_b, coeffs_c)):
    fn = jax.jit(lambda u: coeff_a * jnp.sin(coeff_c * u) + coeff_b * jnp.cos(coeff_c * u))
    grad_fn = jax.jit(lambda u: coeff_a * coeff_c + jnp.cos(coeff_c * u) - coeff_b * coeff_c + jnp.sin(coeff_c * u))
    for j in range(quests):
      while True:
        init = data_utils.generate_gaussian_process(next(rng), xs, num, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (num, N+1)
        if jnp.max(jnp.abs(init)) < 3.0:
          break
      sol = generate_weno_scalar_sol(dx = 1.0 / length, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn, stable_tol = 10.0) # (num, steps + 1, N, 1)
      all_xs.append(xs) # (N,)
      all_us.append(sol) # (num, steps + 1, N, 1)
      all_params.append("{:.8f}_{:.8f}_{:.8f}".format(coeff_a, coeff_b, coeff_c))
      all_eqn_captions.append(['dummy caption'])
    utils.print_dot(i)
    if (i+1) % (len(coeffs_a)//FLAGS.file_split) == 0 or i == len(coeffs_a) - 1:
      for ptype in ['forward', 'backward']:
        for st in FLAGS.stride:
          sti = int(st)
          write_evolution_tfrecord(seed = next(rng)[0], eqn_type = eqn_type, 
                            all_params = all_params, all_eqn_captions = all_eqn_captions,
                            all_xs = all_xs, all_us = all_us, stride = sti,
                            problem_type = ptype, file_name = "{}_{}_{}_stride{}_{}.tfrecord".format(name, eqn_type, ptype, sti, i+1))
      all_xs = []; all_us = []; all_params = []; all_eqn_captions = []


def write_evolution_tfrecord(seed, eqn_type, all_params, all_eqn_captions, all_xs, all_us, stride, problem_type, file_name):
  '''
  xs: (N,)
  us: (num, steps + 1, N, 1)
  '''
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  print("===========" + file_name + "==========", flush=True)
  count = 0
  with tf.io.TFRecordWriter(file_name) as writer:
    for params, eqn_captions, xs, us in zip(all_params, all_eqn_captions, all_xs, all_us):
      equation_name = "{}_{}_{}_{}".format(eqn_type, problem_type, params, stride)
      caption = eqn_captions
      u1 = einshape("ijkl->(ij)kl", us[:,:-stride,:,:]) # (num * step, N, 1)
      u2 = einshape("ijkl->(ij)kl", us[:,stride:,:,:]) # (num * step, N, 1)
      # shuffle the first dimension of u1 and u2, keep the same permutation
      key = next(rng)
      u1 = jax.random.permutation(key, u1, axis = 0) # (num * step, N, 1)
      u2 = jax.random.permutation(key, u2, axis = 0) # (num * step, N, 1)
      # reshape u1 and u2 to (num, s, N, 1)
      u1 = einshape("(ij)kl->ijkl", u1, i = us.shape[0]) # (num, step, N, 1)
      u2 = einshape("(ij)kl->ijkl", u2, i = us.shape[0]) # (num, step, N, 1)
      
      if FLAGS.truncate is not None:
        u1 = u1[:,:FLAGS.truncate,:,:] # (num, truncate, N, 1)
        u2 = u2[:,:FLAGS.truncate,:,:] # (num, truncate, N, 1)

      x1 = einshape("k->jkl", xs, j = u1.shape[1], l = 1) # (truncate or step, N, 1)
      x2 = einshape("k->jkl", xs, j = u2.shape[1], l = 1) # (struncate or step, N, 1)

      if problem_type == 'forward':
        for i in range(us.shape[0]): # split into num parts
          count += 1
          s_element = datawrite.serialize_element(equation = equation_name, caption = caption, 
                                                  cond_k = x1, cond_v = u1[i], qoi_k = x2, qoi_v = u2[i], count = count)
          writer.write(s_element)
      elif problem_type == 'backward':
        for i in range(us.shape[0]):
          count += 1
          s_element = datawrite.serialize_element(equation = equation_name, caption = caption, 
                                                  cond_k = x2, cond_v = u2[i], qoi_k = x1, qoi_v = u1[i], count = count)
          writer.write(s_element)
      else:
        raise NotImplementedError("problem_type = {} is not implemented".format(problem_type))


def test_weno_linear(b, n = 10):
  dir_name = 'linear_plots'
  if not os.path.exists('./{}'.format(dir_name)):
    os.makedirs('./{}'.format(dir_name))
  # test and plot generate_weno_scalar_sol
  xs = jnp.linspace(0.0, 1.0, 100, endpoint=False)
  steps = 1000
  fn = jax.jit(lambda u: b * u)
  grad_fn = jax.jit(lambda u: b + u * 0)
  rng = hk.PRNGSequence(jax.random.PRNGKey(42))
  # init = data_utils.generate_gaussian_process(next(rng), xs, n, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (batch, N, 1)
  init = jnp.exp(- (xs-0.5) ** 2 / 0.2 ** 2)[None,:,None] # (batch, N, 1)
  sol = generate_weno_scalar_sol(dx = 1.0 / 100, dt = 0.001, init = init, fn = fn, steps = steps, grad_fn = grad_fn)
  for i in range(1):
    plt.figure(figsize= (18,7))
    for j in range(0,steps+1,100):
      plt.subplot(2,6,j//100+1)
      # blue line with red markers of size 2
      plt.plot(xs, sol[i, j,:,0], 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('t = {:.3f}\n integral = {:.5f}'.format(j * 0.001, 0.01 * np.sum(sol[i, j,:,0])))
      plt.xlim([0,1])
      plt.ylim([np.min(sol[i,0,:,0])-0.1,np.max(sol[i,0,:,0])+0.1])
    plt.suptitle('du/dt + d({:.2f} u)/dx = 0'.format(b))
    plt.tight_layout()
    plt.savefig('{}/weno_linear_b{:.2f}_{}.pdf'.format(dir_name, b, i))
    # close all figures
    plt.close('all')


def test_weno_burgers(b, n = 10):
  dir_name = 'burgers_plots'
  if not os.path.exists('./{}'.format(dir_name)):
    os.makedirs('./{}'.format(dir_name))
  # test and plot generate_weno_scalar_sol
  xs = jnp.linspace(0.0, 1.0, 100, endpoint=False)
  dx = 0.01
  steps = 2000
  plot_stride = 200
  dt = 0.0005
  fn = jax.jit(lambda u: b * u * u)
  grad_fn = jax.jit(lambda u: 2 * b * u)
  rng = hk.PRNGSequence(jax.random.PRNGKey(42))
  init = jnp.concatenate([jnp.zeros((25,)), jnp.ones((50,)), jnp.zeros((25,))])[None,:,None] # (batch, N, 1)
  sol = generate_weno_scalar_sol(dx = dx, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn)
  for i in range(1):
    plt.figure(figsize= (18,7))
    for j in range(0,steps+1,plot_stride):
      plt.subplot(2,6,j//plot_stride+1)
      # blue line with red markers of size 2
      plt.plot(xs, sol[i, j,:,0], 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('t = {:.3f}\n integral = {:.5f}'.format(j * dt, dx * np.sum(sol[i, j,:,0])))
      plt.xlim([0,1])
      plt.ylim([np.min(sol[i,0,:,0])-0.1,np.max(sol[i,0,:,0])+0.1])
    plt.suptitle('du/dt + d({:.2f} u^2)/dx = 0'.format(b))
    plt.tight_layout()
    plt.savefig('{}/weno_burgers_b{:.2f}_{}.pdf'.format(dir_name, b, "piecewise_1"))
    # close all figures
    plt.close('all')

def test_weno_quadratic(a, b, n = 2):
  dir_name = 'quadratic_plots'
  if not os.path.exists('./{}'.format(dir_name)):
    os.makedirs('./{}'.format(dir_name))
  # test and plot generate_weno_scalar_sol
  xs = jnp.linspace(0.0, 1.0, 100, endpoint=False)
  steps = 1000
  fn = jax.jit(lambda u: a * u * u + b * u)
  grad_fn = jax.jit(lambda u: 2 * a * u + b)
  rng = hk.PRNGSequence(jax.random.PRNGKey(42))
  init = data_utils.generate_gaussian_process(next(rng), xs, 32, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (batch, N, 1)
  sol = generate_weno_scalar_sol(dx = 1.0 / 100, dt = 0.001, init = init, fn = fn, steps = steps, grad_fn = grad_fn)
  for i in range(n):
    plt.figure(figsize= (18,7))
    for j in range(0,1001,100):
      plt.subplot(2,6,j//100+1)
      # blue line with red markers of size 2
      plt.plot(xs, sol[i, j,:,0], 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('t = {:.3f}'.format(j * 0.001))
      plt.xlim([0,1])
      plt.ylim([np.min(sol[i,0,:,0])-0.1,np.max(sol[i,0,:,0])+0.1])
    plt.tight_layout()
    plt.suptitle('du/dt + d({:.2f} u^2 + {:.2f} u)/dx = 0'.format(a, b))
    plt.savefig('{}/weno_burgers_{:.2f}uu_{:.2f}u_{}.pdf'.format(dir_name, a, b, i))
    # close all figures
    plt.close('all')

def test_weno_cubic(a, b, c, n = 2):
  dir_name = 'cubic_plots'
  if not os.path.exists('./{}'.format(dir_name)):
    os.makedirs('./{}'.format(dir_name))
  # test and plot generate_weno_scalar_sol
  xs = jnp.linspace(0.0, 1.0, 100, endpoint=False)
  steps = 1000
  dt = 0.0005
  dx = 1.0 / 100
  fn = jax.jit(lambda u: a * u * u * u + b * u * u + c * u)
  grad_fn = jax.jit(lambda u: 3 * a * u * u + 2 * b * u + c)
  rng = hk.PRNGSequence(jax.random.PRNGKey(42))
  init = data_utils.generate_gaussian_process(next(rng), xs, 32, kernel = data_utils.rbf_circle_kernel_jax, k_sigma = 1.0, k_l = 1.0)[...,None] # (batch, N, 1)
  sol = generate_weno_scalar_sol(dx = dx, dt = dt, init = init, fn = fn, steps = steps, grad_fn = grad_fn)
  for i in range(n):
    plt.figure(figsize= (18,7))
    for j in range(0,steps+1,100):
      plt.subplot(2,6,j//100+1)
      # blue line with red markers of size 2
      plt.plot(xs, sol[i, j,:,0], 'b-', marker='o', markersize=2, markerfacecolor='red', markeredgecolor='red')
      plt.title('t = {:.3f}\n integral = {:.5f}'.format(j * dt, 0.01 * np.sum(sol[i, j,:,0])))
      plt.xlim([0,1])
      plt.ylim([np.min(sol[i,0,:,0])-0.1,np.max(sol[i,0,:,0])+0.1])
    plt.suptitle('du/dt + d({:.2f} u^3 + {:.2f} u^2 + {:.2f} u)/dx = 0'.format(a, b, c))
    plt.tight_layout()
    plt.savefig('{}/weno_cubic_{:.2f}uuu_{:.2f}uu_{:.2f}u_{}.pdf'.format(dir_name, a, b, c, i))
    # close all figures
    plt.close('all')

def main(argv):
  for key, value in FLAGS.__flags.items():
      print(value.name, ": ", value._value, flush=True)
  
  
  name = '{}/{}'.format(FLAGS.dir, FLAGS.name)

  if not os.path.exists(FLAGS.dir):
    os.makedirs(FLAGS.dir)

  if 'test_linear' in FLAGS.eqn_types:
    for b in np.linspace(-1, 1, 3):
      test_weno_linear(b)

  if 'test_burgers' in FLAGS.eqn_types:
    # for b in np.linspace(0.1, 1, 10):
    for b in [0.5]:
      test_weno_burgers(b)
     
  if 'test_quadratic' in FLAGS.eqn_types:
    for a in np.linspace(-1, 1, 5):
      for b in np.linspace(-1, 1, 5):
        print("a = {}, b = {}".format(a, b), flush=True)
        test_weno_quadratic(a, b)
  
  if 'test_cubic' in FLAGS.eqn_types:
    for a in np.linspace(-1, 1, 3):
      for b in np.linspace(-1, 1, 3):
        for c in np.linspace(-1, 1, 3):
          print("a = {}, b = {}, c = {}".format(a, b, c), flush=True)
          test_weno_cubic(a, b, c)

  if 'weno_burgers' in FLAGS.eqn_types:
    generate_conservation_weno_burgers(
          seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, steps = 1000,
          dt = FLAGS.dt, num = FLAGS.num, name = name)
    
  if 'weno_quadratic' in FLAGS.eqn_types:
    generate_conservation_weno_quadratic(
          seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, steps = 1000,
          dt = FLAGS.dt, num = FLAGS.num, name = name, eqn_mode = FLAGS.eqn_mode)
    
  if 'weno_cubic' in FLAGS.eqn_types:
    generate_conservation_weno_cubic(
          seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, steps = 1000,
          dt = FLAGS.dt, num = FLAGS.num, name = name, eqn_mode = FLAGS.eqn_mode)

  if 'weno_sin' in FLAGS.eqn_types:
    generate_conservation_weno_sin(
          seed = FLAGS.seed, eqns = FLAGS.eqns, quests = FLAGS.quests, length = FLAGS.length, steps = 1000,
          dt = FLAGS.dt, num = FLAGS.num, name = name, eqn_mode = FLAGS.eqn_mode)  

if __name__ == "__main__":

  import tensorflow as tf
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  tf.config.set_visible_devices([], device_type='GPU')

  FLAGS = flags.FLAGS
  flags.DEFINE_integer('num', 100, 'number of systems in each equation')
  flags.DEFINE_integer('quests', 1, 'number of questions in each operator')
  flags.DEFINE_integer('eqns', 100, 'number of equations')
  flags.DEFINE_integer('length', 100, 'length of trajectory and control')
  flags.DEFINE_float('dt', 0.001, 'time step in dynamics')
  flags.DEFINE_float('dx', 0.01, 'time step in dynamics')
  flags.DEFINE_string('name', 'data', 'name of the dataset')
  flags.DEFINE_string('dir', '.', 'name of the directory to save the data')
  flags.DEFINE_list('eqn_types', ['weno_quadratic'], 'list of equations for data generation')
  flags.DEFINE_list('write', [], 'list of features to write')

  flags.DEFINE_list('stride', [200], 'time strides')

  flags.DEFINE_integer('seed', 1, 'random seed')
  flags.DEFINE_string('eqn_mode', 'random_-1_1', 'the mode of equation generation')
  flags.DEFINE_integer('file_split', 10, 'split the data into multiple files')
  flags.DEFINE_integer('truncate', None, 'truncate the length of each record')

  app.run(main)
