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
from matplotlib import cm

# detector parameters
npmts = 10
radius = 1.0
absorption_length = 10
scattering_length = 10

# configuration
tracking = False
debug = False
debug_hardcore = False
if debug_hardcore: debug = True

# plotting functions
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
  q = [0] * npmts
  for ev in events:
    for i in range(len(ev.pmts)):
      q[i] += ev.pmts[i].q/len(events)
  f = plt.figure(figsize=(2,2), facecolor='white')
  h = plt.hist(nhit, bins=100, log=False)
  plt.title('Charge distribution')
  plt.xlabel('Total charge in PMT')
  plt.ylabel('Counts')

def plot_pdfs(pdfs, pos=None):
  cols = min(npmts, 5)
  rows = math.ceil(npmts/cols)
  f = plt.figure(figsize=(1,3), facecolor='white')
  title = 'Time vs Charge Distributions'
  f.suptitle(title, fontsize=18)
  for ipdf in range(len(pdfs)):
    pdf = pdfs[ipdf]
    ax = plt.subplot(int(rows), int(cols), int(ipdf+1))
    extent = [pdf.yedges[0], pdf.yedges[-1], pdf.xedges[-1], pdf.xedges[0]]
    ax.imshow(pdf.h, extent=extent, interpolation='nearest', aspect='equal')
    ax.set_title('Time vs charge, PMT ' + str(ipdf), fontsize=12)
    ax.set_xbound(0,5)
    ax.set_ybound(0,2)

def plot_tracks(tx, ty):
  from pylab import linspace
  an = numpy.arange(0, 2 * math.pi, 0.01)
  f = plt.figure(figsize=(2, 2), facecolor='white')
  a = plt.plot(numpy.cos(an), numpy.sin(an), 'b-', tx, ty, 'ro-')
  plt.title('All tracks')
  plt.xlabel('x')
  plt.ylabel('y')

# data classes
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

class Hist2D:
  '''holds a 2d histogram along with bin edges'''
  def __init__(self, h, xe, ye):
    self.h = h
    self.xedges = xe
    self.yedges = ye

# analysis functions
def make_pdfs(events):
  '''creates from events (a list of Events) a list (one per pmt) of Hist2D
  objects (which in turn hold a numpy.histogram2d).
  '''
  n = len(events)
  q = [None] * npmts
  t = [None] * npmts
  for event in events:
    for ipmt in range(len(event.pmts)):
      charge = event.pmts[ipmt].q
      time = event.pmts[ipmt].t
      try:
        q[ipmt].append(charge)
        t[ipmt].append(time)
      except AttributeError:
        q[ipmt] = [charge]
        t[ipmt] = [time]

  pdfs = []
  for ipmt in range(npmts):
    nph = numpy.histogram2d(t[ipmt], q[ipmt], bins=(50,50))
    h = Hist2D(*nph)
    pdfs.append(h)

  return pdfs

# simulation functions
def pick_charge(mu=1.0, sigma=1.0):
  '''returns a gaussian-distributed single photoelectron charge'''
  mu = 1.0
  sigma = 1.0
  q = random.gauss(mu, sigma)
  return max(0.0, q)

def get_hit_pmt_position(position, theta):
  '''given a photon position and angle (in the photon's coordinates), finds the
  position of the pmt it will hit. returns the distance to, angle to, and
  coordinates of that pmt.
  '''
  event_x = position[0]
  event_y = position[1]
  event_r = math.sqrt(event_x**2 + event_y**2)

  a = 2.0 * (event_x*math.cos(theta) + event_y*math.sin(theta))
  ray_len = -a/2 + 0.5 * math.sqrt(a**2 - 4*(event_r**2 - radius**2))
  pmt_x = ray_len*math.cos(theta) + event_x
  pmt_y = ray_len*math.sin(theta) + event_y
  angle_to_pmt = math.atan2(pmt_y, pmt_x)

  return ray_len, (pmt_x, pmt_y), angle_to_pmt

def get_random_position():
  event_r = radius + 1
  while event_r > radius:
    event_x = radius*(2*random.random()-1)
    event_y = radius*(2*random.random()-1)
  return (event_x, event_y)

def simulate(position, theta, photons_per_event, processes):
  '''simulate propagates optical photons to pmts.

  input: position: an (x,y) tuple, the position of the photon bomb

         theta: an angle in radians, the starting momentum of the photon.
                use 'random' for isotropic photons.

         photons_per_event: number of photons to throw at this location, cf.
                            a 'photon bomb'

         processes: a dictionary of the interaction lengths for various photon
                    processes

  output: returns a Event including the simulated position and a list of PMT
          objects (including unhit PMTs).
  '''
  event = Event()
  event_x = position[0]
  event_y = position[1]

  track_x = []
  track_y = []

  for photon in range(photons_per_event):
    pos = (event_x, event_y)
    event_r = math.sqrt(event_x**2 + event_y**2)
    if theta == 'random':
      theta = random.vonmisesvariate(0, 0)

    if debug:
      print 'photon', photon
      print ' start', pos, 'r =', event_r

    if tracking:
      track_x.append(event_x)
      track_y.append(event_y)

    reached_pmt = False
    kill_photon = False
    while not reached_pmt:
      ray_len, pmt_pos, angle_to_pmt = get_hit_pmt_position(pos, theta)

      interaction_lengths = {'pmt': ray_len}
      for process in processes:
        try:
          interaction_lengths[process] = processes[process].pop(0)
        except IndexError:
          print 'Ran out of numbers for process', process
          sys.exit(1)
      process = min(interaction_lengths, key = interaction_lengths.get)

      if process == 'pmt':
        reached_pmt = True

      if process == 'absorption':
        if debug: print 'photon absorbed'
        kill_photon = True
        break

      if process == 'scattering':
        x = pos[0] + interaction_lengths[process] * math.cos(theta)
        y = pos[1] + interaction_lengths[process] * math.sin(theta)
        r = math.sqrt(x**2 + y**2)
        if r > 1: raise ValueError

        if debug: print ' scattered', pos, 'r =', r
        if debug_hardcore:
          print '  abs(ray_len):', abs(ray_len)
          print '  interaction_length:', interaction_lengths[process]
          print '  theta', theta
          print '  pos[0]:', pos[0]
          print '  pos[1]:', pos[1]
          print '  x:', x
          print '  y:', y
          print '  r:', r

        pos = (x, y)
        theta = random.vonmisesvariate(0, 0) # isotropic scattering

        if tracking:
          track_x.append(x)
          track_y.append(y)

        continue

    if kill_photon: continue
 
    pmtid = int(math.floor((angle_to_pmt + math.pi) / (2 * math.pi) * npmts))
    if tracking:
      track_x.append(pmt_pos[0])
      track_y.append(pmt_pos[1])
    if debug: print ' hit pmt', pmtid, math.degrees(angle_to_pmt)

    event.pmts[pmtid].npe += 1
    event.pmts[pmtid].q += 1.0 #max(pick_charge(), 0.0)
    # earliest photon time
    if (ray_len < event.pmts[pmtid].t) or (event.pmts[pmtid].t == 0.0):
      event.pmts[pmtid].t = abs(ray_len)

  return event, track_x, track_y

# main
if __name__ == '__main__':
  processes = {}
  processes['scattering'] = [random.expovariate(1.0 / scattering_length) for i in range(int(1e6))]
  processes['absorption'] = [random.expovariate(1.0 / absorption_length) for i in range(int(1e6))]

  events = []
  for i in range(100):
    if i%10 == 0: print i
    pos = (0.5, 0.0)
    theta = random.vonmisesvariate(0, 0)
    ev, tx, ty = simulate(pos, theta, 100, processes)
    events.append(ev)

    if tracking: plot_tracks(tx,ty)

  pdfs = make_pdfs(events)
  plot_pdfs(pdfs, pos=pos)

  plt.show()

# testing functions
def test_process(process, nphotons):
  processes = {}
  position = (0.75, 0.0)
  for j in [1000, 100, 1, 0.1]:
    print process, 'length =', j, '...'
    processes[process] = [random.expovariate(1.0/j) for i in range(int(1e6))]
    ev, tx, ty = simulate(position, 'random', nphotons, processes)
    plot_hit_distribution_polar(ev, pos = 'lamba=' + str(j))
    f = plt.figure(figsize=(2,2), facecolor='white')
    plt.title('lambda='+str(j))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    p = plt.plot(tx,ty,marker='o')
  plt.show()
  sys.exit(0)

