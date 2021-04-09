import time

import torch
import torchvision


def usage_in_pytorch():
    model = torchvision.models.resnet18(pretrained=True)
    data = torch.rand(1, 3, 64, 64)
    labels = torch.rand(1, 1000)

    prediction = model(data)  # forward pass
    loss = (prediction - labels).sum()
    loss.backward()  # backward pass

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()  # gradient descent


def differentiation_in_autograd():
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)

    Q = 3*a**3 - b**2

    external_grad = torch.tensor([1., 1.])
    Q.backward(gradient=external_grad)

    # check if collected gradients are correct
    print(9 * a ** 2 == a.grad)
    print(-2 * b == b.grad)


def exclusions_from_the_dag():
    x = torch.rand(5, 5)
    y = torch.rand(5, 5)
    z = torch.rand((5, 5), requires_grad=True)

    a = x + y
    print(f"Does `a` require gradients? : {a.requires_grad}")
    b = x + z
    print(f"Does `b` require gradients?: {b.requires_grad}")

    model = torchvision.models.resnet18(pretrained=True)

    # Freeze all the parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(512, 10)
    # Optimize only the classifier
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

    # Optimize only the classifier
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


def main():

    # usage_in_pytorch()
    differentiation_in_autograd()
    exclusions_from_the_dag()


start = time.time()
main()
end = time.time()
total_time = end - start
print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
