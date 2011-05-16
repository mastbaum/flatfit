#!/usr/bin/env python

# flatfit.py
#
# flatfit simulates a circular snoplus-like detector with 100% pmt coverage.
#
# Andy Mastbaum (mastbaum@hep.upenn.edu), May 2011
#

import sys
import math, random
import numpy
from scipy import special as sp
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid

npmts = 10
radius = 1
debug = False
debug_hardcore = False


def pick_charge(mu=1.0, sigma=1.0):
  '''returns a gaussian-distributed single photoelectron charge'''
  mu = 1.0
  sigma = 1.0
  q = random.gauss(mu, sigma)
  return max(0.0, q)

def plot_hit_distribution_polar(hits, pos=None):
  npmts = len(hits)
  fig1 = plt.figure(figsize=(2,2), facecolor='white')
  ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
  theta = numpy.arange(0, 2*math.pi, 2*math.pi/npmts)
  radii = numpy.array(hits)
  width = 2*numpy.pi/npmts
  bars = ax.bar(theta, radii, width=width, bottom=0.0)
  for r,bar in zip(radii, bars):
    bar.set_facecolor( cm.jet(r/float(max(hits))))
    bar.set_alpha(0.5)
  title = 'Distribution of PMT hits'
  if position is not None: title += (', ' + str(pos))
  plt.title(title)

def plot_hit_distribution(hits):
  pmts = numpy.arange(0, len(hits))
  fig2 = plt.figure(figsize=(2,2), facecolor='white')
  bar = plt.bar(pmts, hits)
  plt.title('Distribution of PMT hits')
  plt.xlabel('PMT ID')
  plt.ylabel('NHits')

def histogram_mean_pe(l, pmtid):
  fig3 = plt.figure(figsize=(2,2), facecolor='white')
  plt.title('Charge histogram, PMT ' + str(pmtid))
  plt.xlabel('Charge')
  plt.ylabel('Counts')

def histogram_time(l, pmtid):
  fig3 = plt.figure(figsize=(2,2), facecolor='white')
  h = plt.hist(l, bins=100, log=False)
  plt.title('Timing histogram, PMT ' + str(pmtid))
  plt.xlabel('Time')
  plt.ylabel('Counts')

def histogram_mean_pe_all(q):
  npmts = len(q)
  cols = min(npmts, 5)
  rows = math.ceil(npmts/cols)
  f = plt.figure(figsize=(2,2), facecolor='white')
  f.suptitle('PMT Charge Distributions', fontsize=18)
  plt.subplots_adjust(hspace=0.4)
  i = 0
  while(i < npmts):
    ax = plt.subplot(rows, cols, i+1)
    ax.hist(q[i], bins=20, log=True)
    ax.set_title('Charge dist., PMT ' + str(i), fontsize=12)
    ax.set_xbound(0,2)
    i += 1

def histogram_time_all(t):
  npmts = len(t)
  cols = min(npmts, 5)
  rows = math.ceil(npmts/cols)
  f = plt.figure(figsize=(2,2), facecolor='white')
  f.suptitle('PMT Timing Distributions', fontsize=18)
  plt.subplots_adjust(hspace=0.4)
  i = 0
  while(i < npmts):
    ax = plt.subplot(rows, cols, i+1)
    ax.hist(t[i], bins=20)
    ax.set_title('Time dist., PMT ' + str(i), fontsize=12)
    ax.set_xbound(0,2)
    i += 1

def q_vs_t_all(q, t, pos=None):
  npmts = len(q)
  cols = min(npmts, 5)
  rows = math.ceil(npmts/cols)
  f = plt.figure(figsize=(2,2), facecolor='white')
  title = 'Time vs Charge Distributions'
  if pos is not None: title += (' ' + str(pos))
  f.suptitle(title, fontsize=18)
  #plt.subplots_adjust(hspace=0.4,wspace=0.4)
  i = 0
  while(i < npmts):
    ax = plt.subplot(int(rows), int(cols), int(i+1))
    h, xedges, yedges = numpy.histogram2d(t[i], q[i], bins=(50,50))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    ax.imshow(h, extent=extent, interpolation='nearest', aspect='equal')
    ax.set_title('Charge vs time, PMT ' + str(i), fontsize=12)
    ax.set_xbound(0,2)
    ax.set_ybound(0,2)
    i += 1

def simulate(nevents, pos, photons_per_event):
  # initialize results arrays
  hits = [0] * npmts
  q = []
  t = []
  for i in range(npmts):
    q.append([])
    t.append([])

  # event loop
  for event in range(nevents):
    if (event % 10 == 0):
      print '.',
      sys.stdout.flush()
    if (event == nevents - 1): print
    if debug:
      print '== event',event,'===='
    if position is 'random':
      event_r = radius + 1
      while event_r > radius:
        event_x = radius*(2*random.random()-1)
        event_y = radius*(2*random.random()-1)
        event_r = math.sqrt(event_x**2 + event_y**2)
    else:
      event_x = position[0]
      event_y = position[1]
      event_r = math.sqrt(event_x**2 + event_y**2)

    # loop over this event's photons
    q_temp = []
    t_temp = []
    hits_temp = []
    for i in range(npmts):
      q_temp.append(0.0)
      t_temp.append(0.0)
      hits_temp.append(0)
    for photon in range(photons_per_event):
      if debug: print '-- photon', photon, '----'
      theta = random.vonmisesvariate(0, 0)
      if position == (0.0, 0.0):
        ray_len = radius
        angle_to_pmt = theta - math.pi
      else:
        a = 2.0 * (event_x*math.cos(theta) + event_y*math.sin(theta))
        ray_len = -a/2 - 0.5 * math.sqrt(a**2 - 4*(event_r**2 - radius**2))
        global_x = ray_len*math.cos(theta) + event_x
        global_y = ray_len*math.sin(theta) + event_y
        angle_to_pmt = math.atan2(global_y, global_x)

      if debug_hardcore:
        with open('local','w') as f:
          f.write('%f\t%f\t%f\t%f\n' % (event_x, event_y, ray_len*math.cos(theta), ray_len*math.sin(theta)))
        with open('global','w') as f:
          f.write('%f\t%f\t%f\t%f\n' % (0.0, 0.0, math.cos(angle_to_pmt), math.sin(angle_to_pmt)))
        print 'event', event
        print ' local: ', event_x, event_y, ray_len*math.cos(theta), ray_len*math.sin(theta)
        print ' global:', 0.0, 0.0, ray_len*math.cos(angle_to_pmt), ray_len*math.sin(angle_to_pmt)

      pmt = int(math.floor((angle_to_pmt+math.pi)/(2*math.pi) * npmts))
      #print pmt, ray_len, theta, angle_to_pmt
      if debug:
        print 'pmt:', pmt

      hits_temp[pmt] += 1

      ch = pick_charge()
      if debug:
        print 'q =', ch,
        if ch < 0:
          raise Exception
      q_temp[pmt] += ch
      # first photon time
      if (ray_len < t_temp[pmt]) or (t_temp[pmt] == 0.0):
        if debug: print 't =', abs(ray_len)
        t_temp[pmt] = abs(ray_len)
      else:
        if debug: print

      if debug:
        print q_temp
        print t_temp

    for pmt in range(npmts):
      if hits_temp[pmt] > 0:
        mean_pe = q_temp[pmt]/hits_temp[pmt]
        q[pmt].append(mean_pe)
        t[pmt].append(t_temp[pmt])
        hits[pmt] += hits_temp[pmt]
      else:
        q[pmt].append(0.0)
        t[pmt].append(0.0)

  return hits, q, t

# main
if __name__ == '__main__':
  #position = (-0.75, 0.0)
  #position = 'random'
  nevents = 1000
  photons_per_event = 500 # use fn w energy spectrum

  charge = []
  time = []
  hits = []

  for position in [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]:
    print '**', position, '***'
    h, q, t = simulate(nevents, position, photons_per_event)
    plot_hit_distribution_polar(h, pos=position)
    q_vs_t_all(q, t, pos=position)
    hits.append(h)
    charge.append(q)
    time.append(t)

  # output
  if debug:
    print '** summary ****'
    if debug_hardcore:
      print q
      print t
    print hits

  #histogram_mean_pe_all(q)
  #histogram_time_all(t)

  plt.show()

