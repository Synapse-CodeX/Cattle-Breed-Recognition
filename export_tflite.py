import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_dir", type=str, default="saved_model/breed_classifier")
parser.add_argument("--out_tflite", type=str, default="model.tflite")
parser.add_argument("--quantize", action="store_true",
                    help="Apply post-training float16 or full integer quantization (requires representative data for int8).")
parser.add_argument("--quant_type", type=str, default="float16", choices=["float16", "int8"])
parser.add_argument("--data_dir", type=str, default="/kaggle/input/cattle-breed-recognition/dataset")
parser.add_argument("--representative_samples", type=int, default=100)
args = parser.parse_args()

model = tf.keras.models.load_model(args.saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

if args.quantize:
    if args.quant_type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(args.out_tflite, "wb") as f:
            f.write(tflite_model)
        print("Saved float16 TFLite:", args.out_tflite)
    else:
        def representative_gen():
            ds = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(args.data_dir, "train"),
                image_size=(224,224),
                batch_size=1,
                shuffle=True
            )
            count = 0
            for batch, _ in ds:
                img = tf.cast(batch, tf.float32)
                img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
                yield [img]
                count += 1
                if count >= args.representative_samples:
                    break
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        with open(args.out_tflite, "wb") as f:
            f.write(tflite_model)
        print("Saved int8 TFLite:", args.out_tflite)
else:
    tflite_model = converter.convert()
    with open(args.out_tflite, "wb") as f:
        f.write(tflite_model)
    print("Saved float32 TFLite:", args.out_tflite)

