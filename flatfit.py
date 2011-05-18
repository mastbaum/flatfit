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
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid

npmts = 10
radius = 1.0
attenuation_length = 10
scattering_length = 10
debug = False
debug_hardcore = False


def pick_charge(mu=1.0, sigma=1.0):
  '''returns a gaussian-distributed single photoelectron charge'''
  mu = 1.0
  sigma = 1.0
  q = random.gauss(mu, sigma)
  return max(0.0, q)

def plot_hit_distribution_polar(ev, pos=None):
  '''polar plot of hit distribution for a single event'''
  npmts = len(ev.pmts)
  fig1 = plt.figure(figsize=(2,2), facecolor='white')
  ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
  theta = numpy.arange(0, 2*math.pi, 2*math.pi/npmts)

  # total number of hits in each tube
  hits = [0] * npmts
  for i in range(len(ev.pmts)):
    hits[i] += ev.pmts[i].npe

  radii = numpy.array(hits)
  width = 2*numpy.pi/npmts
  bars = ax.bar(theta, radii, width=width, bottom=0.0)
  for r,bar in zip(radii, bars):
    bar.set_facecolor( cm.jet(r/float(max(hits))))
    bar.set_alpha(0.5)
  title = 'Distribution of PMT hits'
  if pos is not None: title += (', ' + str(pos))
  plt.title(title)

def histogram_nhit(events):
  nevents = len(events)
  npmts = len(events[0].pmts)
  nhit = [0] * len(events)
  for e in range(nevents):
    for p in range(len(ev.pmts)):
      if ev.pmts[p].npe > 0:
        nhit[e] += 1
  # apparent bug in matplotlib.
  # if all values are the same (here 50), the bin is incorrectly labeled as 0.0
  nhit.append(0.0)
  f = plt.figure(figsize=(2,2), facecolor='white')
  h = plt.hist(nhit, bins=100, log=False)
  plt.title('Nhit distribution')
  plt.xlabel('Number of incident photons')
  plt.ylabel('Counts')

def histogram_q(events):
  npmts = len(events[0].pmts)
  nhit = [0] * npmts
  for ev in events:
    for i in range(len(ev.pmts)):
      nhit[i] += ev.pmts[i].q/len(events)
  f = plt.figure(figsize=(2,2), facecolor='white')
  h = plt.hist(nhit, bins=100, log=False)
  plt.title('Charge distribution')
  plt.xlabel('Total charge in PMT')
  plt.ylabel('Counts')

#def histogram_qpe(events):
#  npmts = len(events[0].pmts)
#  nhit = [0] * npmts
#  for ev in events:
#    for i in range(len(ev.pmts)):
#      nhit[i] += (ev.pmts[i].q/ev.pmts[i].npe)/len(events)
#  f = plt.figure(figsize=(2,2), facecolor='white')
#  h = plt.hist(nhit, bins=100, log=False)
#  plt.title('Charge distribution')
#  plt.xlabel('Total charge in PMT')
#  plt.ylabel('Counts')

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

class PMT:
  '''holds a pmt's total charge, hit time, and number of photoelectrons for a
  single event.
  '''
  def __init__(self):
    self.npe = 0
    self.q = 0.0
    self.t = 0.0
  def add_hit(self, nphotons, charge, time):
    self.npe = npe
    self.q = charge
    self.t = time

class Event:
  '''holds data for a single event, including position and a list of PMTS'''
  def __init__(self):
    self.pos = None
    self.pmts = []
    for i in range(npmts):
      self.pmts.append(PMT())

def get_hit_pmt_position(position):
  '''given a photon position, picks a random momentum and find the distance to
  the pmt it will hit, and the angle (in detector coordinates) to that pmt.
  '''
  event_x = position[0]
  event_y = position[1]
  event_r = math.sqrt(event_x**2 + event_y**2)
  theta = random.vonmisesvariate(0, 0)
  if position == (0.0, 0.0, 0.0):
    ray_len = radius
    angle_to_pmt = theta - math.pi
  else:
    a = 2.0 * (event_x*math.cos(theta) + event_y*math.sin(theta))
    ray_len = -a/2 + 0.5 * math.sqrt(a**2 - 4*(event_r**2 - radius**2))
    pmt_x = ray_len*math.cos(theta) + event_x
    pmt_y = ray_len*math.sin(theta) + event_y
    angle_to_pmt = math.atan2(pmt_y, pmt_x)

  return ray_len, theta, angle_to_pmt

def simulate(position, photons_per_event):
  '''simulate propagates optical photons to pmts.

  input: position: either an (x,y) tuple or 'random'. the latter will result in
                   a uniformly-distributed random location inside radius.

         photons_per_event: number of photons to throw at this location, cf.
                            a 'photon bomb'

  output: returns a Event including the simulated position and a list of PMT
          objects (including unhit PMTs).
  '''
  event = Event()
  if position is 'random':
    event_r = radius + 1
    while event_r > radius:
      event_x = radius*(2*random.random()-1)
      event_y = radius*(2*random.random()-1)
  else:
    event_x = position[0]
    event_y = position[1]

  track_x = []
  track_y = []

  # loop over this event's photons
  for photon in range(photons_per_event):
    pos = (event_x, event_y)

    event_r = math.sqrt(event_x**2 + event_y**2)
    if debug: print 'photon', photon
    if debug: print ' start',pos,'r =',event_r
    track_x.append(event_x)
    track_y.append(event_y)

    reached_pmt = False
    kill_photon = False
    while not reached_pmt:
      ray_len, theta, angle_to_pmt = get_hit_pmt_position(pos)
      # scattering (isotropic reemission)
      if abs(ray_len) < random.expovariate(1.0/scattering_length):
        # absorption
        if abs(ray_len) > random.expovariate(1.0/attenuation_length):
          kill_photon = True
          break
        reached_pmt = True
      else:
        scattering_distance = random.uniform(0, abs(ray_len)) # uniform?
        # absorption before it can scatter?
        if abs(scattering_distance) > random.expovariate(1.0/attenuation_length):
          kill_photon = True
          break
        new_x = pos[0] + scattering_distance * math.cos(theta)
        new_y = pos[1] + scattering_distance * math.sin(theta)
        new_r = math.sqrt(new_x**2 + new_y**2)
        if debug: print ' scattered', pos,'r =',new_r
        if debug_hardcore:
          print '  abs(ray_len):',abs(ray_len)
          print '  scattering_dist:',scattering_distance
          print '  theta',theta
          print '  pos[0]:',pos[0]
          print '  pos[1]:',pos[1]
          print '  new_x:',new_x
          print '  new_y:',new_y
          print '  new_r:',new_r
          if new_r > 1: raise Exception
        pos = (new_x, new_y)
        track_x.append(new_x)
        track_y.append(new_y)
 
    if kill_photon:
      if debug: print ' killed photon'
      continue

    pmtid = int(math.floor((angle_to_pmt+math.pi)/(2*math.pi) * npmts))
    if debug: print ' hit pmt', pmtid, math.degrees(angle_to_pmt)

    event.pmts[pmtid].npe += 1

    event.pmts[pmtid].q += 1.0 #max(pick_charge(), 0.0)

    # earliest photon time
    if (ray_len < event.pmts[pmtid].t) or (event.pmts[pmtid].t == 0.0):
      event.pmts[pmtid].t = abs(ray_len)

  return event, track_x, track_y

def spectrum_flat():
  return 1000

def test_scattering():
  position = (0.5, 0.0)
  attenuation_length = 100 * radius
  for j in [100, 1, 0.1, 0.01, 0.001]:
    exec 'scattering_length = ' + str(j) in globals()
    ev, tx, ty = simulate(position, spectrum_flat())
    plot_hit_distribution_polar(ev, pos=position)
  plt.show()

def test_absorption():
  position = (0.5, 0.0)
  for j in [10, 1, 0.5, 0.1, 0.01]:
    exec 'attenuation_length = ' + str(j) in globals()
    ev, tx, ty = simulate(position, spectrum_flat())
    plot_hit_distribution_polar(ev, pos=position)
  plt.show()

# main
if __name__ == '__main__':
  test_scattering()

  #events = []
  #for position in [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]:
  #  print '**', position, '***'
  #  for ev in range(100):
  #    ev, tx, ty = simulate(position, spectrum_flat())
  #    plot_hit_distribution_polar(ev, pos=position)
  #    events.append(ev)

  #histogram_nhit(events)
  #histogram_q(events)
  #histogram_qpe(events)

  #plt.show()

