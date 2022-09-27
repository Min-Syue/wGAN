 # **Wasserstein GAN (wGAN)**
 
## wGAN是什麼?
>在 GAN 中，需要注意其中的 Generator 和 Discriminator 要"同步前進"，意思為如果其中一方訓練得太好(太聰明)，就容易導致不收斂。<br>
>因此在此前提下，wGAN做出了以下四點的修改，讓 GAN 的訓練更加穩定：
> - Discriminator 的輸出層移除 Sigmoid 函數。
> - Discriminator 在訓練過程中，進行權重裁減(weights clip)。
> - 評估模型的損失函數，從 Cross Entropy 修改成 Wasserstein loss。
> - 不使用含有動量(momentum)的最佳化方法，如 Adam，而在此論文中，使用 RMSprop。
> <br>
>最後整體演算法如下：
>![](https://github.com/Min-Syue/wGAN/blob/main/wGAN_alg.PNG)
>圖片取自wGAN的論文，連結在![這裡](https://arxiv.org/abs/1701.07875)
## 生成結果
> ### Mnist 結果
> 每張數字圖片上的 label 代表要模型生成的數字，如第一張圖片(左上角) label 為4，則代表 Generator 要生成數字4的圖片！
>![](https://github.com/Min-Syue/wGAN/blob/main/wGAN_4000epochs_mnist.gif)
