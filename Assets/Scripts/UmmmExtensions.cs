using System;
using System.IO;
using System.Collections.Generic;

namespace Ummm {
    public static class Extensions {
        private static System.Random rng = new System.Random();

        public static int ReadBigInt32(this BinaryReader br) {
            var bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action) {
            for (var w = 0; w < source.GetLength(0); w++) {
                for (var h = 0; h < source.GetLength(1); h++) {
                    action(w, h);
                }
            }
        }
        
        // public static IEnumerable<ValueTuple<TFirst, TSecond>>? Zip<TFirst, TSecond>(this IEnumerable<TFirst> first,
        //     IEnumerable<TSecond> second) {
        //     return first.Zip(second);
        // }
    }
}