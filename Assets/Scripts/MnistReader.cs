using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.IO;
using System.Linq;

public static class MnistReader {
    public class Data {
        public int width;
        public int height;
        public int imageSize;
        public int numberOfImages;
        
        public double[][] xs;
        public int[] ys;
    }
    
    private const string TrainImages = "Assets/Data/training_images_ubyte.bytes";
    private const string TrainLabels = "Assets/Data/training_labels_ubyte.bytes";
    private const string TestImages = "Assets/Data/test_images_ubyte.bytes";
    private const string TestLabels = "Assets/Data/test_labels_ubyte.bytes";

    public static Data ReadAssets(TextAsset imageAsset, TextAsset labelAsset) {
        return Read(new MemoryStream(imageAsset.bytes), new MemoryStream(labelAsset.bytes));
    }

    public static Data ReadTrainingData() {
        return Read(File.OpenRead(TrainImages), File.OpenRead(TrainLabels));
    }

    public static Data ReadTestData() {
        return Read(File.OpenRead(TestImages), File.OpenRead(TestLabels));
    }

    private static Data Read(Stream imagesStream, Stream labelsStream) {
        var labels = new BinaryReader(labelsStream);
        var images = new BinaryReader(imagesStream);

        images.ReadBigInt32(); // unused: magic number

        var numberOfImages = images.ReadBigInt32();
        var width = images.ReadBigInt32();
        var height = images.ReadBigInt32();

        labels.ReadBigInt32(); // unused: magic number
        labels.ReadBigInt32(); // unused: number of labels

        var xs = new double[numberOfImages][];
        var ys = new int[numberOfImages];

        foreach (var i in Enumerable.Range(0, numberOfImages)) {
            var bytes = images.ReadBytes(width * height);
            xs[i] = bytes.ToList().Select(b => b / 255d).ToArray();
            ys[i] = labels.ReadByte();
        }

        var result = new Data() {
            xs = xs,
            ys = ys,
            width = width,
            height = height,
            imageSize = width * height
        };

        images.Dispose();
        labels.Dispose();

        return result;
    }
}