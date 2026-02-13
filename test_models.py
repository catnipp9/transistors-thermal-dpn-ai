"""
Test script to verify both CNN and sklearn models work correctly.
Run this to test your trained models before deployment.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image


def test_cnn_model():
    """Test the CNN model with a sample image."""
    print("\n" + "="*60)
    print("TEST 1: CNN Model (Thermal Images)")
    print("="*60)

    model_path = Path("checkpoints/best_model.pth")

    if not model_path.exists():
        print(f"❌ CNN model not found at {model_path}")
        print("   Run the training notebook first!")
        return False

    try:
        from api.inference import DPNClassifier

        # Load model
        classifier = DPNClassifier(
            model_path=str(model_path),
            model_type="cnn"
        )
        print(f"✅ CNN model loaded successfully")

        # Find test images
        control_images = list(Path("data/Control Group").glob("*/*_L.png"))
        diabetic_images = list(Path("data/DM Group").glob("*/*_L.png"))

        if not control_images or not diabetic_images:
            print("❌ No test images found in data folder")
            return False

        # Test Control sample
        print(f"\n📷 Testing Control sample: {control_images[0].name}")
        result = classifier.predict(str(control_images[0]))
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Probabilities: Control={result['probabilities']['Control']}%, Diabetic={result['probabilities']['Diabetic']}%")

        # Test Diabetic sample
        print(f"\n📷 Testing Diabetic sample: {diabetic_images[0].name}")
        result = classifier.predict(str(diabetic_images[0]))
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Probabilities: Control={result['probabilities']['Control']}%, Diabetic={result['probabilities']['Diabetic']}%")

        print("\n✅ CNN model test PASSED")
        return True

    except Exception as e:
        print(f"❌ CNN model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sklearn_model():
    """Test the sklearn model with a sample CSV."""
    print("\n" + "="*60)
    print("TEST 2: sklearn Model (Temperature Values)")
    print("="*60)

    model_path = Path("checkpoints/best_sklearn_model.joblib")

    if not model_path.exists():
        print(f"❌ sklearn model not found at {model_path}")
        print("   Run the training notebook and save the sklearn model!")
        return False

    try:
        from api.inference import DPNClassifier

        # Load model
        classifier = DPNClassifier(
            model_path=str(model_path),
            model_type="sklearn"
        )
        print(f"✅ sklearn model loaded successfully")

        # Find test CSVs
        control_csvs = list(Path("data/Control Group").glob("*/*_L.csv"))
        diabetic_csvs = list(Path("data/DM Group").glob("*/*_L.csv"))

        if not control_csvs or not diabetic_csvs:
            print("❌ No test CSV files found in data folder")
            return False

        # Test Control sample
        print(f"\n📊 Testing Control sample: {control_csvs[0].name}")
        result = classifier.predict(str(control_csvs[0]))
        print(f"   Prediction: {result['prediction']}")
        if 'confidence' in result:
            print(f"   Confidence: {result['confidence']}%")

        # Test Diabetic sample
        print(f"\n📊 Testing Diabetic sample: {diabetic_csvs[0].name}")
        result = classifier.predict(str(diabetic_csvs[0]))
        print(f"   Prediction: {result['prediction']}")
        if 'confidence' in result:
            print(f"   Confidence: {result['confidence']}%")

        print("\n✅ sklearn model test PASSED")
        return True

    except Exception as e:
        print(f"❌ sklearn model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_predictions():
    """Test predictions on multiple samples to check accuracy."""
    print("\n" + "="*60)
    print("TEST 3: Batch Accuracy Check (CNN)")
    print("="*60)

    model_path = Path("checkpoints/best_model.pth")

    if not model_path.exists():
        print("❌ CNN model not found, skipping batch test")
        return False

    try:
        from api.inference import DPNClassifier

        classifier = DPNClassifier(
            model_path=str(model_path),
            model_type="cnn"
        )

        # Test multiple samples
        correct = 0
        total = 0

        # Test 10 Control samples
        control_images = list(Path("data/Control Group").glob("*/*_L.png"))[:10]
        for img_path in control_images:
            result = classifier.predict(str(img_path))
            if result['prediction'] == 'Control':
                correct += 1
            total += 1

        # Test 10 Diabetic samples
        diabetic_images = list(Path("data/DM Group").glob("*/*_L.png"))[:10]
        for img_path in diabetic_images:
            result = classifier.predict(str(img_path))
            if result['prediction'] == 'Diabetic':
                correct += 1
            total += 1

        accuracy = (correct / total) * 100 if total > 0 else 0

        print(f"\n📊 Results on {total} samples:")
        print(f"   Correct: {correct}/{total}")
        print(f"   Accuracy: {accuracy:.1f}%")

        if accuracy >= 70:
            print("\n✅ Batch test PASSED")
            return True
        else:
            print("\n⚠️ Accuracy lower than expected - model may need more training")
            return False

    except Exception as e:
        print(f"❌ Batch test FAILED: {e}")
        return False


def main():
    print("\n" + "#"*60)
    print("#  DPN CLASSIFICATION MODEL TESTER")
    print("#"*60)

    results = []

    # Run tests
    results.append(("CNN Model", test_cnn_model()))
    results.append(("sklearn Model", test_sklearn_model()))
    results.append(("Batch Accuracy", test_batch_predictions()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    print("\n" + "="*60)

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("🎉 All tests passed! Your models are ready for deployment.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    return all_passed


if __name__ == "__main__":
    main()
