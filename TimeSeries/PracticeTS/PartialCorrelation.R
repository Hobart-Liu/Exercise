# init
setwd('~/exercise/TimeSeries/PracticeTs/')
rm(list=ls(all=TRUE))
while (dev.cur()>1) dev.off()

# dataset
library(isdals)
data(bodyfat)

# help(bodyfat)
# 确定人体内的脂肪是昂贵且麻烦的，因为它涉及将人浸入水中。 
# 该数据集提供了20位年龄在20至34岁的健康女性的体脂，三头肌皮褶厚度，
# 大腿围和手臂中部围的信息。如果模型可以提供可靠的体内脂肪量预测，
# 则是可取的，因为需要进行测量 因为预测变量很容易获得。

# Fat:人体脂肪
# Triceps：三头肌皮褶测量
# Thigh:大腿围
# Midarm:中臂围
detach()
attach(bodyfat)
# attach()
# 用$ 符号访问对象不是非常的方便，如accountants$statef。一个非常有用的工
# 具将会使列表或者数据框的分量可以通过它们的名字直接调用。而且这种调用是暂时
# 性的，没有必要每次都显式的引用列表名字。
pairs( cbind( Fat, Triceps, Thigh, Midarm) )

print(cor( cbind( Fat, Triceps, Thigh, Midarm) ))
# 显然，脂肪和肱三头肌（皮肤褶皱厚度）高度相关，r = 0.8432654。 
# 但是，大腿周长也是如此，r = 0.8780896。


# 由于Triceps和Thigh有明显的相关性，r = 0.9238425，
# 因此我们想知道在控制或“分离”Thigh后，是否可以测量Fat和Triceps的相关性。
# We first try to account  for the  effect of Thigh on both Fat and Triceps 
# by regressing them on Thigh. After we remove the contribution of Thigh, 
# we then find the correlation of Fat and Triceps





