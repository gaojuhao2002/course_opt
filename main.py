
def fun(n,x=1):                  #(n,x)位置参数，x默认参数可以通过实名改变，但是这个应为是嵌套所以只要第一次成功改变
    print('x=', x)
    if n==1:
        return x
    else:
        return fun(n-1)*n
def fun2(*args,x):                #可变参数实际上传入的是元组被绑定了注意无论参数位置在在哪里都可以通过实名制赋值
    print(type(args),type(x))

def fun3(**kwargs):               #传入的是位置参数，字典保存。调用时类似传入kwa=5
    print('other',kwargs)
def test19(x):
    n=6
    y=0
    c=100
    for i in range(int(n/2)):
        y=y+0.5*(x(2*i-2)^2+c*x(2*i-1)^2)
    return y

if __name__ == '__main__':#作用import时候，其他文件不会调用这之后的内容
    a=3//2;#整除
    b=3**2;#指数
    #逻辑运算符 not and or
    if a>b:
        print(a)
    else:
        print([0]*6)
    print(fun(5,6))
    fun2((4,5,6),x=1)
    print(fun3(you=5))
    #函数  不可变传值，可变对象传地址
    a,b=2,
    print('a',a,'b',b)