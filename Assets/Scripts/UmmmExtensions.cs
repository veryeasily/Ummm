using System;
using System.IO;
using System.Collections.Generic;

public static class UmmmExtensions {
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

    public static void Shuffle<T>(this IList<T> list) {
        var n = list.Count;  
        while (n > 1) {  
            n--;  
            var k = rng.Next(n + 1);  
            var value = list[k];  
            list[k] = list[n];  
            list[n] = value;  
        }  
    }
}