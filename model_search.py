import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

def random_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    indices = torch.randperm(num_channels)
    x = x[:,indices]

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)

    for primitive in PRIMITIVES:
      op = OPS[primitive](C //4, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //4, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    random_shuffle(x)
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//4, :, :]
    xtemp2 = x[ : ,  dim_2//4:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)

    return ans


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        # input nodes的结构固定不变，不参与搜索
        # 决定第一个input nodes的结构，取决于前一个cell是否是reduction
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0,
                                          affine=False)  # 第一个input_nodes是cell k-2的输出，cell k-2的输出通道数为C_prev_prev，所以这里操作的输入通道数为C_prev_prev
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)  # 第二个input_nodes是cell k-1的输出
        self._steps = steps  # 每个cell中有4个节点的连接状态待确定
        self._multiplier = multiplier

        self._ops = nn.ModuleList()  # 构建operation的modulelist
        self._bns = nn.ModuleList()
        # 遍历4个intermediate nodes构建混合操作
        for i in range(self._steps):
            # 遍历当前结点i的所有前驱节点
            for j in range(2 + i):  # 对第i个节点来说，他有j个前驱节点（每个节点的input都由前两个cell的输出和当前cell的前面的节点组成）
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)  # op是构建两个节点之间的混合
                self._ops.append(op)  # 所有边的混合操作添加到ops，list的len为2+3+4+5=14[[],[],...,[]]

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]  # 当前节点的前驱节点
        offset = 0
        # 遍历每个intermediate nodes，得到每个节点的output
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j]) for j, h in
                    enumerate(states))  # s为当前节点i的output，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
            offset += len(states)
            states.append(s)  # 把当前节点i的output作为下一个节点的输入
            # states中为[s0,s1,b1,b2,b3,b4] b1,b2,b3,b4分别是四个intermediate output的输出
        return torch.cat(states[-self._multiplier:], dim=1)  # 对intermediate的output进行concat作为当前cell的输出
        # dim=1是指对通道这个维度concat，所以输出的通道数变成原来的4倍


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps  ##有4个节点的连接状态待确定
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i == 2:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # 构建一个平均池化层，output size是1x1
        self.classifier = nn.Linear(C_prev, num_classes)  # 构建一个线性分类器

        self._initialize_alphas()  # 架构参数初始化
        '''
        cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
        layers = 8, 第2和5个cell是reduction_cell
        cells[0]: cell = Cell(4, 4, 48,  48,  16, false,  false) 输出[N,16*4,h,w]
        cells[1]: cell = Cell(4, 4, 48,  64,  16, false,  false) 输出[N,16*4,h,w]
        cells[2]: cell = Cell(4, 4, 64,  64,  32, True,   false) 输出[N,32*4,h,w]
        cells[3]: cell = Cell(4, 4, 64,  128, 32, false,  false) 输出[N,32*4,h,w]
        cells[4]: cell = Cell(4, 4, 128, 128, 32, false,  false) 输出[N,32*4,h,w]
        cells[5]: cell = Cell(4, 4, 128, 128, 64, True,   false) 输出[N,64*4,h,w]
        cells[6]: cell = Cell(4, 4, 128, 256, 64, false,  false) 输出[N,64*4,h,w]
        cells[7]: cell = Cell(4, 4, 256, 256, 64, false,  false) 输出[N,64*4,h,w]
    '''

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                # edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

