def     plot_img_pred(img, pred):
    import matplotlib.pyplot as plt

    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    fig = plt.figure()

    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(pred, cmap='gray')
    plt.axis('off')

    plt.show()


def plot_img_pred_gt(img, pred, gt, title, dsp):
    import matplotlib.pyplot as plt
    plt.subplots_adjust(top=0.85)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    # title += '\r\n' + dsp
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    a = fig.add_subplot(1, 3, 1)
    a.set_title('Input image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    b = fig.add_subplot(1, 3, 2)
    b.set_title('Predicted mask')
    plt.imshow(pred, cmap='gray')
    plt.axis('off')

    c = fig.add_subplot(1, 3, 3)
    c.set_title('Ground Truth mask')
    plt.imshow(gt, cmap='gray')
    plt.axis('off')

    plt.show()
