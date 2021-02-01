using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Ummm {
    public class DatasetObject : ScriptableObject {
        public TextAsset images;
        public TextAsset labels;

        public (double[][], double[]) Data() {
            var foo = new double[784][];
            var bar = new double[784];
            return (foo, bar);
        }
    }
}
