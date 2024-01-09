from getdata import returndata
from torch.nn import init
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from sklearn.decomposition import PCA
from pylab import *
from matplotlib.colors import LinearSegmentedColormap

def min_k(num_list,n):
    pad = max(num_list) + 1
    topn_list = []
    value_list = []
    for i in range(n):
        min_idx = num_list.index(min(num_list))
        value_list.append(min(num_list))
        topn_list.append(min_idx)
        num_list[min_idx] = pad
    return topn_list, value_list

def get_ele_list(the_feature):
    fin_feature = the_feature[6:]
    ele_list = ['Ru', 'Ru']
    for num in fin_feature:
        the_judgements = []
        the_judgements.append(abs(num - 1.2))
        the_judgements.append(abs(num - 1.5))
        the_judgements.append(abs(num - 2.5))
        the_judgements.append(abs(num - 0.5))
        the_judgements.append(abs(num - 1.4))
        indices = sorted(range(len(the_judgements)), key=lambda k: the_judgements[k])
        if indices[0] == 0:
            ele_list.append('Ru')
        if indices[1] == 0:
            ele_list.append('Rh')
        if indices[2] == 0:
            ele_list.append('Ir')
        if indices[3] == 0:
            ele_list.append('Pd')
        if indices[4] == 0:
            ele_list.append('Pt')
    return ele_list

class Network(nn.Module):
    def __init__(self, feat_num):
        super(Network, self).__init__()
        self.layers1str = nn.Linear(3, 5)
        self.layers2str = nn.ReLU()
        self.layers3str = nn.Linear(5, 10)
        self.layers4str = nn.ReLU()

        self.layers1ele = nn.Linear(34, 35)
        self.layers2ele = nn.ReLU()
        self.layers3ele = nn.Linear(35, 10)
        self.layers4ele = nn.ReLU()

        self.layers5 = nn.Linear(20, 1)
        self._init_weight()

        self.z_mu_str = nn.Linear(10, 10)
        self.z_logvar_str = nn.Linear(10, 10)

        self.z_mu_ele = nn.Linear(10, 10)
        self.z_logvar_ele = nn.Linear(10, 10)

        self.layers5str = nn.Linear(10, 5)
        self.layers6str = nn.ReLU()
        self.layers7str = nn.Linear(5, 4)

        self.layers5ele = nn.Linear(10, 35)
        self.layers6ele = nn.ReLU()
        self.layers7ele = nn.Linear(35, 33)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, val=0)

    def forward(self, x):
        x_str = x[:, :3]
        x_ele = x[:, 3:]
        x_str = self.layers1str(x_str)
        x_str = self.layers2str(x_str)
        x_str = self.layers3str(x_str)
        x_str = self.layers4str(x_str)

        x_ele = self.layers1ele(x_ele)
        x_ele = self.layers2ele(x_ele)
        x_ele = self.layers3ele(x_ele)
        x_ele = self.layers4ele(x_ele)
        ele_sampled_z_str, KLD1 = self.reparameterization(self.z_mu_str(x_str), self.z_logvar_str(x_str))
        ele_sampled_z_ele, KLD2 = self.reparameterization(self.z_mu_str(x_ele), self.z_logvar_str(x_ele))
        ele_sampled_z = torch.cat((ele_sampled_z_str, ele_sampled_z_ele), dim=1)
        predict = self.layers5(ele_sampled_z)

        x_simu_str = self.layers5str(ele_sampled_z_str)
        x_simu_str = self.layers6str(x_simu_str)
        x_simu_str = self.layers7str(x_simu_str)

        x_simu_ele = self.layers5ele(ele_sampled_z_ele)
        x_simu_ele = self.layers6ele(x_simu_ele)
        x_simu_ele = self.layers7ele(x_simu_ele)

        simu_x = torch.cat((x_simu_str, x_simu_ele), dim=1)

        return predict, ele_sampled_z_str, ele_sampled_z_ele, simu_x

    def reparameterization(self, mu, log_var):
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
        return mu + sigma * eps, KLD

def run_nn(redir, batch_size):
    if not os.path.isdir('./image/' + redir):
        os.mkdir('./image/' + redir)

    batch_size = batch_size
    use_gpu = torch.cuda.is_available()
    xdata, ydata, label = returndata()
    ydata = np.array([[y] for y in ydata])
    y_data = ydata
    xdata = torch.Tensor(xdata)
    ydata = torch.Tensor(ydata)

    train_set = TensorDataset(xdata, ydata)

    train_loader = DataLoader(train_set,
                              batch_size = batch_size,
                              shuffle=False)

    model = torch.load('./1066_epoch.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    predict_list = []
    z_str_list = []
    z_ele_list = []
    simu_x_list = []
    for x, y in train_loader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        predict, ele_sampled_z_str, ele_sampled_z_ele, simu_x = model(x)
        for item in predict.detach().numpy():
            predict_list.append(list(item))

        for item in ele_sampled_z_str.detach().numpy():
            z_str_list.append(list(item))

        for item in ele_sampled_z_ele.detach().numpy():
            z_ele_list.append(list(item))

        for item in simu_x.detach().numpy():
            simu_x_list.append(list(item))

    z_str_list_pca = PCA(1, random_state = 86).fit_transform(z_str_list)
    z_ele_list_pca = PCA(1, random_state = 86).fit_transform(z_ele_list)
    coor = []
    for i in range(len(z_ele_list_pca)):
        the_coor = [z_str_list_pca[i][0], z_ele_list_pca[i][0]]
        coor.append(the_coor)

    # raw
    plt.figure(figsize=(8, 6.5))
    scatter = plt.scatter(z_str_list_pca, z_ele_list_pca, c=predict_list, s=2.5, lw=1.5, cmap='viridis', edgecolor=None)
    plt.colorbar(scatter)
    plt.title('PCA')
    plt.savefig('./image/' + redir + 'get_space_raw.png', dpi=1000, transparent=True)
    #plt.show()
    
    # class1
    # ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
    z_list = []
    for labe in label:
        if labe[1] == 'ML_HEA_100_bridge':
            z_list.append([6])
        elif labe[1] == 'ML_HEA_110_bridge':
            z_list.append([5])
        elif labe[1] == 'ML_HEA_111_bridge':
            z_list.append([9])
        elif labe[1] == 'ML_HEA_211_edge':
            z_list.append([2])
        elif labe[1] == 'ML_HEA_211_hill':
            z_list.append([11])
        elif labe[1] == 'ML_HEA_211_summit':
            z_list.append([8])
        elif labe[1] == 'ML_HEA_211_valley':
            z_list.append([13])
        elif labe[1] == 'ML_HEA_532_higher_edge':
            z_list.append([4])
        elif labe[1] == 'ML_HEA_532_hill1':
            z_list.append([10])
        elif labe[1] == 'ML_HEA_532_hill2':
            z_list.append([7])
        elif labe[1] == 'ML_HEA_532_inner_higher_edge':
            z_list.append([3])
        elif labe[1] == 'ML_HEA_532_outer_higher_edge':
            z_list.append([1])
        elif labe[1] == 'ML_HEA_532_valley2':
            z_list.append([12])
        else:
            print('>_<')

    #clist = ['lightgreen', 'lightgreen', 'darkorange', 'olive', 'tan', 'darkgreen', 'lightseagreen', 'darkslategrey', 'steelblue', 'slategray', 'navy', 'indigo', 'purple']
    clist = ['#7700BB', '#5500DD', '#4400CC', '#0000CC', '#0044BB', '#009FCC', '#00DDDD', '#00DDAA', '#00DD77', '#00DD00', '#66DD00', '#99DD00', '#EEEE00']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist, N = 13)

    plt.figure(figsize=(8, 6.5))
    scatter = plt.scatter(z_str_list_pca, z_ele_list_pca, c=z_list, s=2.5, lw=1.5, cmap=newcmp, edgecolor=None)
    plt.colorbar(scatter)
    plt.title('PCA')
    plt.savefig('./image/' + redir + 'get_space_class1.png', dpi=1000, transparent=True)
    #plt.show()

    # class2
    z_list = []
    for labe in label:
        if labe[0] == 'Ru-Ru':  # 2.4
            z_list.append([1])
        elif labe[0] == 'Ru-Rh':  # 2.7
            z_list.append([3])
        elif labe[0] == 'Ru-Ir':  # 3.7
            z_list.append([2])
        elif labe[0] == 'Ru-Pd':  # 1.7
            z_list.append([7])
        elif labe[0] == 'Ru-Pt':  # 2.6
            z_list.append([5])
        elif labe[0] == 'Rh-Rh':  # 3
            z_list.append([6])
        elif labe[0] == 'Rh-Ir':  # 4
            z_list.append([4])
        elif labe[0] == 'Rh-pd':  # 2
            z_list.append([9])
        elif labe[0] == 'Rh-Pt':  # 2.9
            z_list.append([12])
        elif labe[0] == 'Ir-Ir':  # 5
            z_list.append([8])
        elif labe[0] == 'Ir-Pd':  # 3
            z_list.append([10])
        elif labe[0] == 'Ir-Pt':  # 3.9
            z_list.append([11])
        elif labe[0] == 'Pd-Pd':  # 1
            z_list.append([15])
        elif labe[0] == 'Pd-Pt':  # 1.9
            z_list.append([13])
        elif labe[0] == 'Pt-Pt':  # 2.8
            z_list.append([14])
        else:
            print('>_<')

    clist = ['#7700BB', '#5500DD', '#4400CC', '#0000CC', '#0044BB', '#009FCC', '#00DDDD', '#00DDAA', '#00DD77',
             '#00DD00', '#66DD00', '#99DD00', '#EEEE00']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist, N=15)
    plt.figure(figsize=(8, 6.5))
    scatter = plt.scatter(z_str_list_pca, z_ele_list_pca, c=z_list, s=2.5, lw=1.5, cmap=newcmp, edgecolor=None)
    plt.colorbar(scatter)
    plt.title('PCA')
    plt.savefig('./image/' + redir + 'get_space_class2.png', dpi=1000, transparent=True)
    #plt.show()

    # result
    z_list = []
    num = 0
    for labe in label:
        if labe[0] == 'Ru-Ru' and labe[1] == 'ML_HEA_532_outer_higher_edge':
            z_list.append([0])
            num += 1
        else:
            z_list.append([1])

    clist = ['#7700BB', '#EEEE00']
    newcmp = LinearSegmentedColormap.from_list('chaos', clist, N=2)
    plt.figure(figsize=(8, 6.5))
    scatter2 = plt.scatter(z_str_list_pca, z_ele_list_pca, c=z_list, s=2.5, lw=1.5, cmap=newcmp, edgecolor=None)
    scatter1 = plt.scatter(z_str_list_pca, z_ele_list_pca, c=z_list, s=2.5, lw=1.5, cmap=newcmp, edgecolor=None, alpha = 0.3)
    plt.colorbar(scatter2)
    plt.title('PCA')
    plt.savefig('./image/' + redir + 'get_space_result.png', dpi=1000, transparent=True)
    #plt.show()

    # get_result
    predict_get_list = []
    for re in predict_list:
        predict_get_list.append(re[0])
    list_num, value_list = min_k(predict_get_list, 8)
    the_po_HEAs = []
    for the_num in list_num:
        the_po_HEAs.append(simu_x_list[the_num])

    the_result_HEA_ele_list = []
    for the_feature in the_po_HEAs:
        ele_list = get_ele_list(the_feature)
        the_result_HEA_ele_list.append(ele_list)
    return the_result_HEA_ele_list

def vae_main(args):
    batch_size = args.batch_size
    for n in range(1):
        redir = str(n) + './'
        list_num = run_nn(redir, batch_size)
        np.save('./get_result_HEA/result_potential_HEA.npy', list_num)

def parse_args(args):
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()
    return args

def do_main():
    args = parse_args(sys.argv[1:])
    vae_main(args)

if __name__ == "__main__":
    do_main()