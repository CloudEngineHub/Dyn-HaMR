#!/usr/bin/env python3
"""
Test script to verify VIPE integration with Dyn-HaMR
"""
import os
import sys
import numpy as np

# Add dyn-hamr to path
sys.path.insert(0, 'dyn-hamr')

def test_vipe_loading():
    """Test loading VIPE cameras"""
    print("="*60)
    print("Testing VIPE Camera Loading")
    print("="*60)
    
    from data.vidproc import load_vipe_cameras
    
    vipe_dir = 'third-party/vipe/vipe_results'
    seq_name = 'prod1'
    img_dir = 'test/images/prod1'
    
    # Check if VIPE files exist
    pose_path = os.path.join(vipe_dir, 'pose', f'{seq_name}.npz')
    intrins_path = os.path.join(vipe_dir, 'intrinsics', f'{seq_name}.npz')
    
    if not os.path.exists(pose_path):
        print(f"✗ VIPE pose file not found: {pose_path}")
        return False
    if not os.path.exists(intrins_path):
        print(f"✗ VIPE intrinsics file not found: {intrins_path}")
        return False
    
    print(f"✓ Found VIPE pose file: {pose_path}")
    print(f"✓ Found VIPE intrinsics file: {intrins_path}")
    
    # Load VIPE cameras
    try:
        w2c, intrins = load_vipe_cameras(vipe_dir, seq_name, img_dir, start=0, end=10)
        print(f"\n✓ Successfully loaded VIPE cameras")
        print(f"  - Loaded {len(w2c)} camera poses")
        print(f"  - w2c shape: {w2c.shape}")
        print(f"  - intrins shape: {intrins.shape}")
        print(f"  - Image size: {int(intrins[0, 4])}x{int(intrins[0, 5])}")
        print(f"  - Focal length: {intrins[0, 0]:.2f}")
        return True
    except Exception as e:
        print(f"\n✗ Error loading VIPE cameras: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vipe_saving():
    """Test saving VIPE cameras in DROID format"""
    print("\n" + "="*60)
    print("Testing VIPE Camera Saving")
    print("="*60)
    
    from data.vidproc import load_vipe_cameras, save_vipe_cameras_as_droid
    
    vipe_dir = 'third-party/vipe/vipe_results'
    seq_name = 'prod1'
    img_dir = 'test/images/prod1'
    output_dir = 'test/vipe_test_output'
    
    try:
        # Load cameras
        w2c, intrins = load_vipe_cameras(vipe_dir, seq_name, img_dir, start=0, end=10)
        
        # Save in DROID format
        save_vipe_cameras_as_droid(output_dir, w2c, intrins)
        
        # Verify saved file
        saved_file = os.path.join(output_dir, 'cameras.npz')
        if os.path.exists(saved_file):
            print(f"\n✓ Successfully saved cameras to: {saved_file}")
            
            # Load and verify
            data = np.load(saved_file)
            print(f"  - Saved keys: {list(data.keys())}")
            print(f"  - w2c shape: {data['w2c'].shape}")
            print(f"  - intrins shape: {data['intrins'].shape}")
            print(f"  - Image size: {int(data['width'])}x{int(data['height'])}")
            print(f"  - Focal length: {float(data['focal']):.2f}")
            
            # Clean up test output
            import shutil
            shutil.rmtree(output_dir)
            print(f"  - Cleaned up test output directory")
            
            return True
        else:
            print(f"\n✗ Failed to save cameras")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_format_compatibility():
    """Test that VIPE format matches DROID-SLAM format"""
    print("\n" + "="*60)
    print("Testing Format Compatibility")
    print("="*60)
    
    from data.dataset import load_cameras_npz
    from data.vidproc import load_vipe_cameras, save_vipe_cameras_as_droid
    import tempfile
    import shutil
    
    vipe_dir = 'third-party/vipe/vipe_results'
    seq_name = 'prod1'
    img_dir = 'test/images/prod1'
    
    try:
        # Load and save VIPE cameras
        w2c, intrins = load_vipe_cameras(vipe_dir, seq_name, img_dir, start=0, end=10)
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp()
        save_vipe_cameras_as_droid(temp_dir, w2c, intrins)
        
        # Load using Dyn-HaMR's function
        camera_path = os.path.join(temp_dir, 'cameras.npz')
        cam_R, cam_t, intrins_loaded, width, height = load_cameras_npz(camera_path, 10)
        
        print(f"\n✓ Successfully loaded VIPE cameras using Dyn-HaMR's load function")
        print(f"  - cam_R shape: {cam_R.shape}")
        print(f"  - cam_t shape: {cam_t.shape}")
        print(f"  - intrins shape: {intrins_loaded.shape}")
        print(f"  - Image size: {width}x{height}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"  - Format is compatible with Dyn-HaMR!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("VIPE Integration Test Suite")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("VIPE Loading", test_vipe_loading()))
    results.append(("VIPE Saving", test_vipe_saving()))
    results.append(("Format Compatibility", test_format_compatibility()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ All tests passed! VIPE integration is working correctly.")
        print("\nYou can now use VIPE by running:")
        print("  cd dyn-hamr")
        print("  python run.py data=video_vipe data.seq=prod1")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

