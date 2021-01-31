using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

public class DigitImage {
    public byte[] pixels;
    public byte label;

    private static string testImagesPath =
        @"/Volumes/Samsung_X5/singing/Unity/Ummm/Assets/Data/test_images_ubyte.bytes";

    private static string testLabelsPath =
        @"/Volumes/Samsung_X5/singing/Unity/Ummm/Assets/Data/test_labels_ubyte.bytes";

    public static List<DigitImage> LoadTestData() {
        var ifsImages = new FileStream(testImagesPath, FileMode.Open);
        var ifsLabels = new FileStream(testLabelsPath, FileMode.Open);

        var brLabels = new BinaryReader(ifsLabels);
        var brImages = new BinaryReader(ifsImages);
        var magic1 = brImages.ReadInt32(); // discarded
        var numImages = brImages.ReadInt32();
        var numRows = brImages.ReadInt32();
        var numColumns = brImages.ReadInt32();
        
        var magic2 = brLabels.ReadInt32(); // discarded
        var numLabels = brLabels.ReadInt32();

        var images = new List<DigitImage>();
        
        for (var imageIdx = 0; imageIdx < 10000; ++imageIdx) {
            var label = brLabels.ReadByte();
            var pixels = brImages.ReadBytes(28 * 28);
            
            images.Add(new DigitImage(pixels, label));
        }
        
        return images;
    }

    public DigitImage(byte[] pixels, byte label) {
        this.pixels = pixels;
        this.label = label;
    }
}