
# 基于gluon的变分自编码

最近在变分自动编码器（variation autoencoders， VAE）上取得的进展使得它成为构建深度生成网络最受欢迎的框架之一。在本文中，我们首先将介绍一些必要的背景知识。然后我们将在论文[贝叶斯变分自动编码](https://arxiv.org/abs/1312.6114)的基础上构建一个VAE模型，并将其应用在MNIST数据集上进行表示学习和样本生成任务。在本文的复现中，我们将使用MXNet的Gluon API，即`gluon.HybridBlock`和`autograd`等。

## 变分自动编码器（VAE）介绍

### 最大期望算法（EM）的快速回顾
通过构建众所周知的最大期望算法（Expectation-Maximization，EM）可以更为快速地了解VAE。请参考这篇[教程](https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf)或者这篇[博客](https://dingran.github.io/algorithm/EM/)来复习EM算法。在这里我们只回顾一下EM算法中的一些关键点：EM算法构造并优化通常被称之为**Evidence Lower Bond (EBLO)**的下界$\mathcal{L}(q,\theta)$，而不是利用数据$x$和隐变量$z$去优化最大似然$\ell(\theta)$。下面的等式可以从Jensen不等式中推导出来，它适用于任意$q(z)$只要它是一个有效的概率分布。

$$
\begin{equation*}
\begin{split}
\ell(\theta) &= \sum_i \log\left( p_{\theta}(x_i) \right) \\
& = \sum_i \log\left( \int p_{\theta}(x_i, z) dz \right)\\
&= \sum_i \log\left( \mathbb{E}_{z \sim Q} \left[ \frac {p_{\theta}(x_i, z)}{q(z)} \right]\right)\\
& {\ge}\underbrace{ \sum_i \mathbb{E}_{z \sim Q} \left[\log\left( \frac {p_{\theta}(x_i,z)}{q(z)} \right)\right]}_{ELBO: \mathcal{L}(q,\theta)}
\end{split}
\end{equation*}
$$

重要的是，在$q(z)$的所有选择中，一旦$q$作为推断后验被选中，我们就能最大化$q$对应的ELBO $\mathcal{L}(q,\theta)$，例如在第t次迭代时，$q^t(z) = p(z\vert x_i; \hat\theta^{t-1}) = \frac{p(x_i\vert z; \hat\theta^{t-1})p(z; \hat\theta^{t-1})}{\int p(x_i\vert z; \hat\theta^{t-1})p(z; \hat\theta^{t-1}) dz}$。这是EM算法中E步的本质。然后在M步中，我们最大化$\theta$。在E步中选择的特定的$q(z)$确保了在EM算法中ELBO $\mathcal{L}(q,\theta)$将单调增加，然后最大似然$\ell(\theta)$也将单调增加。通过E步和M步的迭代改进关系如下所示：

$$\ell(\theta^{t-1}) \underset{E-step}{=} \mathcal L(q^t,\theta^{t-1}) \underset{M-step}{\le} \mathcal L(q^t,\theta^t) \underset{Jensen}{\le} \ell(\theta^{t})$$

### 从EM到VAE

对于更复杂的分布$p_\theta(x\vert z)$，想要通过E步骤中的积分**准确推理**出后验分布$p_\theta(z\vert x)$是非常困难的。这个后验推理的问题可以通过**变分推理**来解决，例如平均场近似（假设$q(z)$为可分解的）或者基于采样的方法（例如，用于求解LDA文档主题生成模型的吉布斯采样）。然而平均场近似为变分族$q(z)$施加了不必要的约束，而基于采样的方法则有着收敛慢的问题。另外，两种方法在进行函数更新时都面临着很大的问题，即便是模型的微小改动也需要重新进行推导，因此这就限制了对更复杂模型的探索。

[贝叶斯变分自动编码](https://arxiv.org/abs/1312.6114)带来了一种基于神经网络的灵活实现。在此框架中，变分推理/变分优化任务（寻找最优$q$）变成了通过反向传播和随机梯度下降寻找神经网络最有参数的问题。因而，这使得黑盒推理成为可能，同时可以对其进行扩展用以训练更深更大的神经网络模型。我们将这种框架称为**神经变分推理**。

下面是它的工作原理：
- 为隐变量$p_\theta(z)$选择先验值，其中可能包含参数，也可能不包含。
- 使用神经网络来参数化分布$p_\theta(x\vert z)$。因为模型的这一部分将隐变量$z$映射为可观测的数据$x$，因此它被视为一个解码器网络。
- 与其显式地计算棘手的$p(z\vert z)$，不如使用另一个神经网络来参数化分布$q_\phi(z\vert x)$作为后验近似。由于能将数据$x$映射为隐变量$z$，因此模型的这一部分被视为一个编码器网络。
- 此时的目标仍然是最大化ELBO $\mathcal{L}(\phi,\theta)$。但现在不像EM算法那样分别寻找最优的$\phi$（对应为EM中的$q$）和$\theta$，我们可以通过标准的随机梯度下降同时找到$\theta$和$\phi$。

这样的模型在结构上类似于编码器-解码器，因此被称为**变分自动编码（variational auto-encoder，VAE）**。

在[贝叶斯变分自动编码](https://arxiv.org/abs/1312.6114)的经典示例中，我们把$p(z)$当做是各向同性的标准高斯分布$\mathcal N(0, I)$，把$q_\phi(z\vert x)$也当做是各向同性的高斯分布$\mathcal N(\mu_\phi(x), \sigma_\phi(x) I)$，其中$\mu_\phi(x)$和 $\sigma_\phi(x)$都是使用神经网络实现的函数，而它们的输出则被当做高斯分布$q_\phi(z\vert x)$的参数。这种模型配置通常被称为**高斯VAE**.

通过这种设置，ELBO的负值可以作为训练损失来优化，其表达式如下：

$$
\begin{equation*}
\begin{split}
- \mathcal L(x_i, \phi,\theta) & = - \mathbb{E}_{z \sim Q_\phi(z|x_i)} \left[\log p_{\theta}(x_i \vert z) + \log p_\theta(z) - \log q_\phi(z\vert x_i) \right] \\
& = - \mathbb{E}_{z \sim Q_\phi(z|x_i)} \left[\log p_{\theta}(x_i \vert z) \right] + D_{KL}\left[\log q_\phi(z\vert x_i) \| p_\theta(z)\right] \\
& \approx \underbrace{\frac{1}{L} \sum_s^L \left[-\log p_{\theta}(x_i \vert z_s) \right]}_{\text{Sampling}\  z_s \sim Q_\phi(z\vert x_i)} + \underbrace{D_{KL}\left[\log q_\phi(z\vert x_i) \| p_\theta(z)\right]}_{\text{Can be calculated analytically between Gaussians}}
\end{split}
\end{equation*}
$$

上式中的ELBO和EM当中的ELBO相同，但是$p(x,z)$增加了$D_{KL}$来表示KL散度，即$D_{KL}(Q \| P)= \mathbb{E}_{x\sim Q}[\log(\frac{q(x)}{p(x)}]$。如公式所示，第一项可以通过从$q_\phi(z\vert x)$分布中抽取$L$蒙特卡洛样本来近似(从各向同性的高斯分布中进行样本抽取是一件相对简单的任务)；而$D_{KL}$可以方便地得到分析解，这是一个比蒙特卡洛采样更好的方法，因为它可以获得更低的方差梯度。

在涉及到采样的情况下，剩下的问题就是我们该如何通过计算图当中的采样节点进行反向传播。[变分贝叶斯自编码](https://arxiv.org/abs/1312.6114)一文的作者采用了一种叫做**Reparameterize Trick (RT)**的技巧。我们不再从$\mathcal N(\mu_\phi(x), \sigma_\phi(x) I)$ 中采样以得到$z$，而是选择从固定分布$\mathcal{N}(0,I)$中采样得$\epsilon$，再通过 $z = \mu(x) + \sigma(x) \cdot \epsilon$构造$z$。这种随机采样的方式依赖于$\epsilon$，因而$z$依赖于$\mu(x)$和$\sigma(x)$，这就使得梯度可以正常通过它们进行传播。RT是一种适用于分布的通用技巧，通过它可以实现位置范围的变换或者CDFs的逆向分析。

***KL散度公式***

$$
D_{KL} (P || Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx
$$

# 高斯VAE复现

既然搞定了理论上的事情，让我们开始复现一下VAE模型吧。


```python
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm, tqdm_notebook
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
def gpu_exists():
    try:
        mx.nd.zeros((1,), ctx=mx.gpu(0))
    except:
        return False
    return True

data_ctx = mx.cpu()
if gpu_exists():
    print('Using GPU for model_ctx')
    model_ctx = mx.gpu(0)
else:
    print('Using CPU for model_ctx')
    model_ctx = mx.cpu()
```


```python
mx.random.seed(1)
output_fig = False
```

## 加载MNIST数据集


```python
mnist = mx.test_utils.get_mnist()
#print(mnist['train_data'][0].shape)
#plt.imshow(mnist['train_data'][0][0],cmap='Greys')

n_samples = 10
idx = np.random.choice(len(mnist['train_data']), n_samples) # 随机获取n_samples张图片
_, axarr = plt.subplots(1, n_samples, figsize=(16,4))
for i,j in enumerate(idx):
    axarr[i].imshow(mnist['train_data'][j][0], cmap='Greys')
    #axarr[i].axis('off')
    axarr[i].get_xaxis().set_ticks([])
    axarr[i].get_yaxis().set_ticks([])
plt.show()
```


```python
train_data = np.reshape(mnist['train_data'],(-1,28*28)) # reshape to 28*28
test_data = np.reshape(mnist['test_data'],(-1,28*28))
```


```python
mnist['test_label'].shape
```


```python
batch_size = 100
n_batches = train_data.shape[0]/batch_size
train_iter = mx.io.NDArrayIter(data={'data': train_data}, label={'label': mnist['train_label']}, batch_size = batch_size)
test_iter = mx.io.NDArrayIter(data={'data': test_data}, label={'label': mnist['test_label']}, batch_size = batch_size)

```

## 模型定义


```python
class VAE(gluon.HybridBlock):
    def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784, batch_size=100, act_type='relu', **kwargs):
        self.soft_zero = 1e-10
        self.n_latent = n_latent
        self.batch_size = batch_size
        self.output = None
        self.mu = None
        # note to self: requring batch_size in model definition is sad, not sure how to deal with this otherwise though
        super(VAE, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        with self.name_scope():
            # 编码器：784--> (400)*n--> 2*2
            self.encoder = nn.HybridSequential(prefix='encoder')
            for i in range(n_layers):
                self.encoder.add(nn.Dense(n_hidden, activation=act_type))
            self.encoder.add(nn.Dense(n_latent*2, activation=None)) # 最后一层无激活
            # 解码器：2*2--> (400)*n--> 784
            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(nn.Dense(n_hidden, activation=act_type))
            self.decoder.add(nn.Dense(n_output, activation='sigmoid')) # 黑白二值图像，因此需要sigmoid激活

    def hybrid_forward(self, F, x):
        '''
        前向计算过程返回的值为损失编码解码过程中的损失大小，而编码的结果则通过net的属性来获取；
        即通过net.mu得到编码的mu，通过net.output得到重建的图像；
        模型训练完成后，也可以通过net.encoder或者net.decoder单独进行编码和解码的工作。
        '''
        h = self.encoder(x) # 编码
        #print(h)
        mu_lv = F.split(h, axis=1, num_outputs=2) # 拆分成两部分
        mu = mu_lv[0]　# mu均值
        lv = mu_lv[1] # lv方差
        self.mu = mu
        # 显式定义了一下eps的shape，以保证其正确工作
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx) # 0中心，1标准差正态分布
        '''
        此处之所以对mu和lv进行了一个公式变换，实际上是因为在VAE中显式的约束了编码的类型为高斯分布，而神经网络前向传播的结果并不能/
        在形式上得到一个高斯分布，因此需要和一个高斯分布进行结合才能得到一个符合高斯分布的编码。
        而这里变换所使用的高斯分布是标准高斯分布，同时也就保证了函数的连续性。
        '''
        z = mu + F.exp(0.5*lv)*eps # 新数据分布，z = mu + exp(lv/2) * eps
        y = self.decoder(z) # 解码
        self.output = y # 存储结果

        KL = 0.5*F.sum(1+lv-mu*mu-F.exp(lv),axis=1) # KL散度计算，具体推导过程可参考其他的文章
        logloss = F.sum(x*F.log(y+self.soft_zero)+ (1-x)*F.log(1-y+self.soft_zero), axis=1) 
        # x*log(y) + (1-x)*log(1-y) # 为何没有选择使用欧氏距离作为loss
        # 此处添加的soft_zero有何用处？
        loss = -logloss-KL # loss加和
        
        return loss
```


```python
n_hidden=400
n_latent=2
n_layers=2 # num of dense layers in encoder and decoder respectively
n_output=784
model_prefix = 'vae_gluon_{}d{}l{}h.params'.format(n_latent, n_layers, n_hidden)

net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size)
```

## 模型训练


```python
net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})
```


```python
n_epoch = 50
print_period = n_epoch // 10
start = time.time()

training_loss = []
validation_loss = []
for epoch in tqdm_notebook(range(n_epoch), desc='epochs'):
    epoch_loss = 0
    epoch_val_loss = 0
    
    train_iter.reset()
    test_iter.reset()
    
    # train
    n_batch_train = 0
    for batch in train_iter:
        n_batch_train +=1
        data = batch.data[0].as_in_context(model_ctx)
        with autograd.record():
            loss = net(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()
    # val
    n_batch_val = 0
    for batch in test_iter:
        n_batch_val +=1
        data = batch.data[0].as_in_context(model_ctx)
        loss = net(data)
        epoch_val_loss += nd.mean(loss).asscalar()
    
    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val
    
    training_loss.append(epoch_loss)
    validation_loss.append(epoch_val_loss)
    # print损失信息
    if epoch % max(print_period,1) == 0:
        tqdm.write('Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))
        
end = time.time()
print('Time elapsed: {:.2f}s'.format(end - start))
```


```python
net.save_params(model_prefix)
```


```python
# 绘制训练过程中损失信息变化情况
batch_x = np.linspace(1, n_epoch, len(training_loss)) # 返回均匀的x坐标
plt.plot(batch_x, -1*np.array(training_loss))
plt.plot(batch_x, -1*np.array(validation_loss))
plt.legend(['train', 'valid'])
```

## 模型加载


```python
net2 = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size)
net2.load_params(model_prefix, ctx=model_ctx)
```

## 可视化重建质量

***VAE功能之一：新数据生成***


```python
test_iter.reset()
test_batch = test_iter.next()
net2(test_batch.data[0].as_in_context(model_ctx))
result = net2.output.asnumpy() # 获取重建结果的方式
original = test_batch.data[0].asnumpy()
```


```python
n_samples = 10
idx = np.random.choice(batch_size, n_samples)
_, axarr = plt.subplots(2, n_samples, figsize=(16,4))
for i,j in enumerate(idx):
    # origin images
    axarr[0,i].imshow(original[j].reshape((28,28)), cmap='Greys')
    if i==0:
        axarr[0,i].set_title('original')
    #axarr[0,i].axis('off')
    axarr[0,i].get_xaxis().set_ticks([])
    axarr[0,i].get_yaxis().set_ticks([])
    # reconstruction images
    axarr[1,i].imshow(result[j].reshape((28,28)), cmap='Greys')
    if i==0:
        axarr[1,i].set_title('reconstruction')
    #axarr[1,i].axis('off')
    axarr[1,i].get_xaxis().set_ticks([])
    axarr[1,i].get_yaxis().set_ticks([])
plt.show()
# 可以看出VAE生成的图片相对原始图像都较为模糊
```

## 可视化隐空间（2D状态）

***VAE的功能之一：数据可视化***


```python
n_batches = 10
counter = 0
results = []
labels = []
for batch in test_iter:
    net2(batch.data[0].as_in_context(model_ctx)) # 写成_ = net2(batch.data[0].as_in_context(model_ctx))可能更好
    results.append(net2.mu.asnumpy())
    labels.append(batch.label[0].asnumpy())
    counter +=1
    if counter >= n_batches:
        break
```


```python
result= np.vstack(results)
labels = np.hstack(labels)
```


```python
if result.shape[1]==2:
    from scipy.special import ndtri
    from scipy.stats import norm
    # 将编码后数字对应的mu分布散点图绘制在2D map上
    fig, axarr = plt.subplots(1,2, figsize=(10,4))
    im=axarr[0].scatter(result[:, 0], result[:, 1], c=labels, alpha=0.6, cmap='Paired')
    axarr[0].set_title(r'scatter plot of $\mu$')
    axarr[0].axis('equal')
    fig.colorbar(im, ax=axarr[0])
    # 使用norm.cdf()函数进行处理，得到累积分布值
    im=axarr[1].scatter(norm.cdf(result[:, 0]), norm.cdf(result[:, 1]), c=labels, alpha=0.6, cmap='Paired')
    axarr[1].set_title(r'scatter plot of $\mu$ on norm.cdf() transformed coordinates')
    axarr[1].axis('equal')
    fig.colorbar(im, ax=axarr[1])
    plt.tight_layout()
    if output_fig:
        plt.savefig('2d_latent_space_for_test_samples.png')
```

## 隐空间采样和图片的生成

### 随机采样


```python
n_samples = 10
zsamples = nd.array(np.random.randn(n_samples*n_samples, n_latent))
```


```python
# 也可以通过直接调用net中的encoder和decoder作为函数进行编码和解码计算
images = net2.decoder(zsamples.as_in_context(model_ctx)).asnumpy() 
```


```python
# 将随机数据生成的图片显示在一张figure中
canvas = np.empty((28*n_samples, 28*n_samples))
for i, img in enumerate(images):
    x = i // n_samples
    y = i % n_samples
    canvas[(n_samples-y-1)*28:(n_samples-y)*28, x*28:(x+1)*28] = img.reshape(28, 28)
plt.figure(figsize=(4, 4))        
plt.imshow(canvas, origin="upper", cmap="Greys")
plt.axis('off')
plt.tight_layout()
if output_fig:
    plt.savefig('generated_samples_with_{}D_latent_space.png'.format(n_latent))
```

### 网格扫描2D隐空间


```python
if n_latent==2: 
    n_pts = 20

    idx = np.arange(0, n_pts)

    x = np.linspace(norm.cdf(-3), norm.cdf(3),n_pts) # 正态分布3sigma之间的等差积分面积序列
    x = ndtri(x) # 对应积分面积下的x值
    # repmat:repeat matrix # meshgrid:网格评估相关的函数，生成一个2D的规律变化矩阵
    x_grid = np.array(np.meshgrid(*[i for i in np.matlib.repmat(x,n_latent,1)])) 
    id_grid = np.array(np.meshgrid(*[i for i in np.matlib.repmat(idx,n_latent,1)]))
    
    # z_samples--> images--> plot
    zsamples = nd.array(x_grid.reshape((n_latent, -1)).transpose()) 
    zsamples_id = id_grid.reshape((n_latent, -1)).transpose()
    images = net2.decoder(zsamples.as_in_context(model_ctx)).asnumpy()

    # 绘图
    canvas = np.empty((28*n_pts, 28*n_pts))
    for i, img in enumerate(images):
        x, y = zsamples_id[i]
        canvas[(n_pts-y-1)*28:(n_pts-y)*28, x*28:(x+1)*28] = img.reshape(28, 28)
    plt.figure(figsize=(6, 6))        
    plt.imshow(canvas, origin="upper", cmap="Greys")
    plt.axis('off')
    plt.tight_layout()
    if output_fig:
        plt.savefig('2d_latent_space_scan_for_generation.png')
    # 从图中可以看出，隐空间中不同区域的mu对应着不同的数字，这一分布和数字的2D可视化是相呼应的。
```
