import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk
from core import *

window = tk.Tk()
window.title('自动牙片匹配系统')  # 设置窗口的标题
window.geometry('1240x400')  # 设置窗口的大小

# 必须先创建全局变量然后更新图片：https://blog.csdn.net/sailist/article/details/79459185
# 创建全局变量impath代表待识别的图片绝对路径,im代表这个路径指向的待匹配图片,matchlist是查询数据库返回的列表,matchim是显示在右边大图框的图片
impath = None
im = None
matchim = None
matchlist = None

# 在窗口中创建4个frame，东南西北各一个
frm_left = tk.LabelFrame(window, text='待匹配图片', padx=10, pady=38)
frm_right = tk.LabelFrame(window, text='匹配图片', padx=10, pady=20)
frm_top = tk.LabelFrame(window, text='程序说明', labelanchor='nw', padx=5, pady=15)
frm_bottom = tk.LabelFrame(window, text='备选图片', labelanchor='n', padx=10, pady=30)
# 设置框架的相对位置，东南西北
frm_left.pack(side='left')
frm_right.pack(side='right')
frm_top.pack(side='top')
frm_bottom.pack(side='bottom')

# 说明性文字的label，修改text即可修改显示文字
# 匹配到若干相似度较高的图片\n最相似的一幅将显示在右侧并注明标签与相似度
infolabel = tk.Label(frm_top,
                     text='程序使用说明：\n点击程序左侧的打开图片\n选择待匹配的JPG图片\n程序会显示该图片\n之后点击右侧的匹配图片\n等待数秒后即可匹配\n点击下方的备选图片还能看到其他图片的情况',
                     # 标签的文字
                     width=40, height=8,  # 标签长宽
                     font=('微软雅黑', 10)
                     )
infolabel.pack(side="top")

# 在底部框架上创建3个label(提前声明，否则后面的绘制bottom frame函数开头的销毁label报错)
label_1 = tk.Label(frm_bottom)
label_2 = tk.Label(frm_bottom)
label_3 = tk.Label(frm_bottom)


# 在左侧的画布上根据图片路径更新图片
def drawleftimg(path):
    global impath
    global im
    impath = path
    img0 = Image.open(path)
    img1 = img0.resize((410, 220))
    im = ImageTk.PhotoImage(img1)
    left_canvas.create_image(0, 40, anchor='nw', image=im)


# 在右侧画布上根据传入的图片路径显示图片
def drawrightimg(path):
    global matchim
    img0 = Image.open(path)
    img1 = img0.resize((410, 220))
    matchim = ImageTk.PhotoImage(img1)
    right_canvas.create_image(0, 40, anchor='nw', image=matchim)


# 默认显示第一张图片的函数
def drawfirst():
    global matchlist
    drawrightimg(matchlist[0]['path'])
    # 两个label显示更新
    si_text.set('标签:{var1}，相似度:{var2}'.format(var1=matchlist[0]['label'], var2=matchlist[0]['similarity']))


# 在点击第1个候选图片时调用，用来显示匹配到的列表的第1个的图片的两个标签信息（label和相似度）和绘制图片
def changeto1(event):
    drawfirst()


# 在点击第2个候选图片时调用，用来显示匹配到的列表的第2个的图片的两个标签信息（label和相似度）和绘制图片
def changeto2(event):
    global matchlist
    drawrightimg(matchlist[1]['path'])
    # 两个label显示更新
    si_text.set('标签:{var1}，相似度:{var2}'.format(var1=matchlist[1]['label'], var2=matchlist[1]['similarity']))


# 在点击第3个候选图片时调用，用来显示匹配到的列表的第3个的图片的两个标签信息（label和相似度）和绘制图片
def changeto3(event):
    global matchlist
    drawrightimg(matchlist[2]['path'])
    # 两个label显示更新
    si_text.set('标签:{var1}，相似度:{var2}'.format(var1=matchlist[2]['label'], var2=matchlist[2]['similarity']))


# 点击打开图片按钮时的调用的函数
def clickleft():
    # 设置可以选择的文件类型，不属于这个类型的，无法被选中
    filetypes = [("JPG文件", "*.jpg")]
    file_name = filedialog.askopenfilename(title='选择待识别图片',
                                           filetypes=filetypes,
                                           initialdir='./'  # 打开当前程序工作目录
                                           )
    file_name = file_name.replace("/", "\\\\")
    drawleftimg(file_name)
    print(impath)


# 点击右边的按钮的对应调用函数
def clickright():
    global impath
    global matchlist
    matchlist = query(impath)
    drawbottom(matchlist)


# 绘制备选匹配图片区域
def drawbottom(lst):
    print(lst)
    lenth = len(lst)
    print("返回匹配个数：" + str(lenth))
    drawfirst()
    # 销毁之前的匹配列表
    label_1.pack_forget()
    label_2.pack_forget()
    label_3.pack_forget()
    # 根据返回数列的长度尽量显示前面元素的图片
    if lenth >= 1:
        # 绘制第1个label，绑定第1个label到更新label1图片的函数
        match1_0 = Image.open(lst[0]['path'])
        match1_1 = match1_0.resize((95, 50))
        match1 = ImageTk.PhotoImage(match1_1)
        label_1.config(image=match1)
        # 保留对Tk对象的引用,防止python的垃圾回收机制将图片变透明
        # 详见：http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
        label_1.image = match1

        label_1.bind('<Button-1>', changeto1)
        label_1.pack(padx=5, pady=5, side=LEFT)

        if lenth >= 2:
            # 绘制第2个label，绑定第2个label到更新label2图片的函数
            match2_0 = Image.open(lst[1]['path'])
            match2_1 = match2_0.resize((95, 50))
            match2 = ImageTk.PhotoImage(match2_1)
            label_2.config(image=match2)
            label_2.image = match2
            label_2.bind('<Button-1>', changeto2)
            label_2.pack(padx=5, pady=5, side=LEFT)

            if lenth >= 3:
                # 绘制第3个label，绑定第3个label到更新label3图片的函数
                match3_0 = Image.open(lst[2]['path'])
                match3_1 = match3_0.resize((95, 50))
                match3 = ImageTk.PhotoImage(match3_1)
                label_3.config(image=match3)
                label_3.image = match3
                label_3.bind('<Button-1>', changeto3)
                label_3.pack(padx=5, pady=5, side=LEFT)
    else:
        # 提示匹配错误框
        messagebox.showwarning('错误', '匹配失败！')


# 打开图片的按钮，font可以自定义字体，width和height是与字的边距
openimgbtn = tk.Button(frm_left, text='打开图片', width=10, height=1, font=('楷体', 16, 'bold'), command=clickleft)
openimgbtn.pack(side="top")
# 识别的按钮
openimgbtn = tk.Button(frm_right, text='点击匹配', width=10, height=1, font=('楷体', 16, 'bold'), command=clickright)
openimgbtn.pack(side="top")

# 初始化相似度标签
si_text = tk.StringVar()  # 创建变量
si_text.set("标签:暂无，相似度:暂无")
label_si = tk.Label(frm_right,
                    width=30, height=2,  # 标签长宽
                    textvariable=si_text
                    )
label_si.pack(side="bottom")  # 将标签固定在窗口上

# 初始化左边的画布
left_canvas = tk.Canvas(frm_left,
                        # bg='blue',       # 设置背景色
                        height=300,  # 设置高度
                        width=410)  # 设置宽度
left_canvas.pack(side="bottom")
imgpath0 = r".\init.jpg"
drawleftimg(imgpath0)
# 初始化右边的画布
right_canvas = tk.Canvas(frm_right,
                         # bg='blue',       # 设置背景色
                         height=300,  # 设置高度
                         width=410)  # 设置宽度
right_canvas.pack()

# 开启GUI自动更新
window.mainloop()
