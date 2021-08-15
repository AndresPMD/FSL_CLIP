import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import json
from utils.data_tools_clip import TaskSampler, EasySet
from utils.utils import sliding_average
from matplotlib import pyplot as plt
import torchvision
import clip


class PrototypicalNetworks(nn.Module):
    def __init__(self, model_clip):
        super(PrototypicalNetworks, self).__init__()
        self.model_clip = model_clip

        self.bn1 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 64)

    def forward(
            self,
            support_images: torch.Tensor,
            support_labels: torch.Tensor,
            query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_support = self.model_clip.encode_image(support_images)
        z_query = self.model_clip.encode_image(query_images)

        z_support = F.relu(self.fc1(self.bn1(z_support.float())))
        z_query = F.relu(self.fc1(self.bn1(z_query.float())))

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat([z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])

        # Compute the euclidean distance from queries to prototypes

        dists = torch.cdist(z_query, z_proto)

        # Distances into classification scores
        scores = -dists
        return scores

def main():
    def fit(support_images, support_labels, query_images, query_labels):
        """
        Args:
            Torch.tensor
        Returns:
            loss as float
        """
        optimizer.zero_grad()
        classification_scores = model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
        loss = criterion(classification_scores, query_labels.cuda())
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate_on_one_task(support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor,
                             query_labels: torch.Tensor):
        """
        Returns the number of correct predictions of query labels, and the total number of predictions.
        """
        return (
                       torch.max(model(support_images.cuda(), support_labels.cuda(), query_images.cuda()).detach().data,
                                 1, )[1]
                       == query_labels.cuda()).sum().item(), len(query_labels)

    def evaluate(data_loader: DataLoader):
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        model.eval()
        with torch.no_grad():
            for episode_index, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    class_ids,
                    #         ) in tqdm(enumerate(data_loader), total=len(data_loader)):
            ) in enumerate(data_loader):
                correct, total = evaluate_on_one_task(support_images, support_labels, query_images, query_labels)

                total_predictions += total
                correct_predictions += correct

        print(
            f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions / total_predictions):.2f}%"
        )
        return correct_predictions / total_predictions * 100

    N_WAY = 8  # Number of classes in a task
    K_SHOT = 5  # Number of images per class in the support set
    N_QUERY = 5  # Number of images per class in the query set
    N_EVALUATION_TASKS = 100

    # EPISODIC TRAINING

    N_TRAINING_EPISODES = 10000
    LEARNING_RATE = 0.0001

    log_update_frequency = 10
    val_episode = 1000


    # Generate Train, Val and Test sets

    train_set = EasySet("./data/train.json", image_size=224, training=True)
    test_set = EasySet("./data/test.json", image_size=224, training=False)
    val_set = EasySet("./data/val.json", image_size=224, training=False)

    # Prototypical Network with ViT/16 pretrained from CLIP as backbone

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_clip, preprocess = clip.load("RN50x16", device=device)
    model_clip, preprocess = clip.load("ViT-B/16", device=device)
    for param in model_clip.parameters():
        param.requires_grad = False

    model = PrototypicalNetworks(model_clip).cuda()
    print("Model Created... Running on: ", device)



    # The sampler needs a dataset with a "labels" field. Check the code if you have any doubt!
    # test_set.labels = [instance[1] for instance in test_set._flat_character_images]
    test_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # Validation Loader
    val_sampler = TaskSampler(
        test_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
    )

    val_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )


    train_sampler = TaskSampler(train_set, n_way=N_WAY, n_shot=K_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # Train the model

    best_accuracy = 0

    all_loss = []
    model.train()

    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            loss_value = fit(support_images, support_labels, query_images, query_labels)
            all_loss.append(loss_value)

            if episode_index % log_update_frequency == 0:
                tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

            if episode_index % val_episode == 0 and episode_index != 0:
                accuracy = evaluate(val_loader)
                tqdm_train.set_postfix(acc=accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), "./models/new_model_ViT.pt")
    print("\nTraining Complete\n!")

    model.load_state_dict(torch.load("./models/new_model_ViT.pt"))
    evaluate(test_loader)

    print("\nAccuracy Per Classes\n!")
    episodes = 50
    model.eval()
    class_names = os.listdir("./data/coco_crops_few_shot/test/")
    accuracies = {class_name: {'total': 0, 'correct': 0} for class_name in class_names}

    for episode in tqdm(range(episodes)):

        (example_support_images, example_support_labels, example_query_images,
         example_query_labels, example_class_ids) = next(iter(test_loader))

        example_scores = model(example_support_images.cuda(), example_support_labels.cuda(),
                               example_query_images.cuda()).detach()

        _, example_predicted_labels = torch.max(example_scores.data, 1)

        for i in range(len(example_query_labels)):

            if test_set.class_names[example_class_ids[example_query_labels[i]]] == test_set.class_names[
                example_class_ids[example_predicted_labels[i]]]:
                accuracies[test_set.class_names[example_class_ids[example_predicted_labels[i]]]]['correct'] += 1
            accuracies[test_set.class_names[example_class_ids[example_predicted_labels[i]]]]['total'] += 1


    for k, v in accuracies.items():
        print("Accuracy per class {} is {:2f}%".format(k, (v['correct'] / v['total']) * 100))

if __name__ == "__main__":
    main()