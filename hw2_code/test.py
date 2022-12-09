"""
@author:rollingball
@time:2022/11/23

"""


def test_dataset():
    from dataset import HW2TrainDataset, HW2TestDataset, HW2RNNTrainDataset, HW2RNNTestDataset
    # train_dataset = HW2TrainDataset()
    # print(len(train_dataset))
    #
    # test_dataset = HW2TestDataset()
    # print(len(test_dataset))

    Rnn_train_dataset = HW2RNNTrainDataset()
    print(len(Rnn_train_dataset))

    Rnn_test_dataset = HW2RNNTestDataset()
    print(len(Rnn_test_dataset))


def test_net():
    from net.net import HW2_net, HW2RNN_net

    model = HW2_net(11 * 39, 41)
    Rnn_model = HW2RNN_net()

    print(model)
    print(Rnn_model)
