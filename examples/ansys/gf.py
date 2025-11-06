import imageio

images = []

for i in range(30):
    filename = f"tmp_img/mesh_optim_{i:03d}.png"
    images.append(imageio.imread(filename))
    print(f"Added {filename} to gif.")
imageio.mimsave("mesh_optim.gif", images, fps=10)
