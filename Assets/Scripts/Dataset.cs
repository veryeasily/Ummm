using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace Ummm {
    public class Dataset {
        public int Width;
        public int Height;
        public int ImageSize;
        public int NumberOfImages;

        public double[][] Xs;
        public int[] Ys;

        private readonly Random _random = new Random();

        public void Shuffle() {
            for (var i = 0; i < Xs.Length; i++) {
                var rand = _random.Next(0, Xs.Length);
                
                var xVal = Xs[i];
                Xs[i] = Xs[rand];
                Xs[rand] = xVal;
                
                var yVal = Ys[i];
                Ys[i] = Ys[rand];
                Ys[rand] = yVal;
            }
        }

        public List<(Vector<double>, double)> GetMiniBatch(int miniBatch, int miniBatchSize) {
            var xs = Xs.Skip(miniBatch * miniBatchSize).Take(miniBatchSize).ToList();
            var ys = Ys.Skip(miniBatch * miniBatchSize).Take(miniBatchSize).ToList();

            return xs.Select((t, i) => (Vector<double>.Build.DenseOfArray(t), ys[i]))
                .Select(dummy => ((Vector<double>, double)) dummy).ToList();
        }

        public static Dataset ReadAssets(byte[] imageBytes, byte[] labelBytes) {
            var imageStream = new MemoryStream(imageBytes);
            var labelStream = new MemoryStream(labelBytes);
            return Read(imageStream, labelStream);
        }

        private static Dataset Read(Stream imagesStream, Stream labelsStream) {
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

            var result = new Dataset() {
                Xs = xs,
                Ys = ys,
                Width = width,
                Height = height,
                ImageSize = width * height
            };

            images.Dispose();
            labels.Dispose();

            return result;
        }
    }
}