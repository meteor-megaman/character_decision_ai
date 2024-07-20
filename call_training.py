from training import train


def main():
    # model_type = 'simple_perceptron'
    model_type = 'multi_layer_perceptron'

    img_str = train(model_type)
    print(img_str)

if __name__ == '__main__':
    main()