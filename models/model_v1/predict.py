import torch
import pandas as pd
from model_v1.dataloader import  TestLettuceDataset
from model_v1.model import model_v1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_RGB = '../datasets/Test/RGBImages'
TEST_DEPTH = '../datasets/Test/DepthImages'
TEST_CSV = '../datasets/Test/Test.csv'
MODEL_PATH = 'best_model.pth'
IMAGE_SIZE = 64
OUTPUT_CSV = 'Test_with_predictions.csv'

def predict_and_save():
	# Load test dataset
	test_dataset = TestLettuceDataset(
		RGB_dir=TEST_RGB,
		depth_dir=TEST_DEPTH,
		image_size=IMAGE_SIZE
	)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

	# Load model
	model = model_v1(num_classes=4).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()

	predictions = {}
	with torch.no_grad():
		for images, image_ids in test_loader:
			images = images.to(DEVICE)
			_, _, fusion_out = model(images)
			preds = fusion_out.cpu().numpy().flatten().tolist()
			# image_ids is a batch
			if isinstance(image_ids, torch.Tensor):
				image_ids = image_ids.cpu().numpy().tolist()
			for img_id, pred in zip(image_ids, preds):
				predictions[int(img_id)] = pred

	# Read original test CSV
	df = pd.read_csv(TEST_CSV)
	# Add predictions in the same order as test.csv
	df['DryWeightShoot'] = df['image_id'].map(lambda x: predictions.get(int(x), ''))
	df.to_csv(OUTPUT_CSV, index=False)
	print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
	predict_and_save()
