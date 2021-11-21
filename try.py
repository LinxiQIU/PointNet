import numpy as np
from path import Path
import os
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import model
import plotly.graph_objects as gf

path = Path("ModelNet10/ModelNet10")

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
classes = {folder: i for i, folder in enumerate(folders)}


# print(classes)

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise 'Not a valid OFF file!'
    nverts, nfaces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for vert in range(nverts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for face in range(nfaces)]
    return verts, faces


with open(path/"bed/train/bed_0001.off", 'r') as f:
    verts, faces = read_off(f)

i, j, k = np.array(faces).T
x, y, z = np.array(verts).T

# print(len(x))


# def visualize_rotation(data):
#     x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
#     frames = []
#
#     def rotate_z(x, y, z, theta):
#         w = x + 1j * y
#         return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
#
#     for t in np.arange(0, 10.26, 0.1):
#         xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
#         frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
#     fig = gf.Figure(data=data,
#                     layout=gf.Layout(updatemenus=[dict(type='button',
#                                                        showactive=False,
#                                                        y=1,
#                                                        x=0.8,
#                                                        xanchor='left',
#                                                        yanchor='bottom',
#                                                        pad=dict(t=45, r=10),
#                                                        buttons=[dict(label='Play', method='animate', args=[
#                                                            None, dict(frame=dict(duration=50, redraw=True),
#                                                                       transition=dict(duration=0),
#                                                                       fromcurrent=True,
#                                                                       mode='immediate')])])]), frames=frames)
#     return fig
#
# visualize_rotation([gf.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.5, i=i, j=j, k=k)]).show()

class PointSample(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros(len(faces))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]))

        sampled_faces = (random.choices(faces, weights=areas, cum_weights=None, k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        return sampled_points


pointcloud = PointSample(3000)((verts, faces))


class Normalize(object):
    def __call__(self, pointcloud):
        # assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return norm_pointcloud


norm_pcd = Normalize()(pointcloud)


class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        return torch.from_numpy(pointcloud)


pcd = ToTensor()(norm_pcd)
# print(pcd)


def default_transforms():
    return transforms.Compose([PointSample(1024), Normalize(), ToTensor()])


class pcddata(Dataset):
    def __init__(self, root_dir, valid=False, folder='train', transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}


train_dataset = pcddata(path, transform=default_transforms())
valid_dataset = pcddata(path, valid=True, folder='test', transform=train_dataset)
inv_classes = {i: cat for cat, i in train_dataset.classes.items()}
# print(inv_classes)
# print('Train dataset size: ', len(train_dataset))
# print('Valid dataset size: ', len(valid_dataset))
# print('Number of classes: ', len(train_dataset.classes))
# print('Sample pointcloud shape: ', train_dataset[0]['pointcloud'].size())
# print('Class: ', inv_classes[train_dataset[0]['category']])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=32)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

pointnet = model.PointNet()
pointnet.to(device)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)


def PointNetLoss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64))/float(bs)


def train(model, train_loder, valid_loader=None, epochs=15, save=True):
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))
            loss = PointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' % (epoch + 1, i+1, len(train_loader), running_loss / 10))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        if valid_loader:
            with torch.no_grad():
                for data in valid_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    __, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Validation accuracy: %d %%' % val_acc)

        if save:
            torch.save(pointnet.state_dict(), "save_"+str(epoch)+".pth")


train(pointnet, train_loader, valid_loader, save=False)
# pointnet.load_state_dict(torch.load('save.pth'))
# pointnet.eval()

