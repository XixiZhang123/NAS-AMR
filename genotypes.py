from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


PC_DARTS_BPSK  = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
BPSKfu4db    = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))
balei         =Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
erbalei =   Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
BALEI = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
sixlayers= Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
twocell = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
twocells = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 1), ('avg_pool_3x3', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
XX = Genotype(normal=[('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3), ('skip_connect', 3), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 3), ('skip_connect', 0)], reduce_concat=range(2, 6))
x102 = Genotype(normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
x103 = Genotype(
  normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1),
          ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6),
  reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0),
          ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
hunhe = Genotype(
  normal=[('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 3),
          ('skip_connect', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=range(2, 6),
  reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 1), ('dil_conv_5x5', 2),
          ('avg_pool_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
x8= Genotype(
  normal=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2),
          ('max_pool_3x3', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 0)], normal_concat=range(2, 6),
  reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('skip_connect', 1),
          ('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))
x2 = Genotype(
  normal=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 3),
          ('dil_conv_5x5', 0), ('avg_pool_3x3', 4), ('sep_conv_3x3', 3)], normal_concat=range(2, 6),
  reduce=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1),
          ('max_pool_3x3', 3), ('dil_conv_5x5', 4), ('skip_connect', 0)], reduce_concat=range(2, 6))
xfu8222 = Genotype(
    normal=[('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3),
            ('dil_conv_3x3', 2), ('max_pool_3x3', 4), ('avg_pool_3x3', 3)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0),
            ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
x10138 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
x10142 = Genotype(
    normal=[('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('avg_pool_3x3', 1),
            ('avg_pool_3x3', 3), ('max_pool_3x3', 4), ('avg_pool_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0),
            ('dil_conv_5x5', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
x6mednodes4 = Genotype(
    normal=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 4), ('sep_conv_3x3', 0), ('dil_conv_5x5', 5), ('sep_conv_3x3', 0),
            ('max_pool_3x3', 6), ('sep_conv_3x3', 1)], normal_concat=range(2, 8),
    reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('skip_connect', 0),
            ('dil_conv_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 5), ('skip_connect', 0),
            ('max_pool_3x3', 5), ('sep_conv_5x5', 0)], reduce_concat=range(2, 8))
x616481 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0),
            ('avg_pool_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 2)], reduce_concat=range(2, 6))
x6161284 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 4), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))
# PCDARTS = twocells
# PCDARTS = x102
# PCDARTS = x103
# PCDARTS = XX
# PCDARTS = x8
# PCDARTS = x2
# PCDARTS = x616481
PCDARTS = x6161284


