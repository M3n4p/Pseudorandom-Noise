using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static HashFunctions;
using static Unity.Mathematics.math;

public class HashVisualization : MonoBehaviour
{
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor
    {
        [WriteOnly]
        public NativeArray<uint> hashes;

        public int resolution;

        public float invResolution;

        public SmallXXHash hash;

        public void Execute(int i)
        {
            float vf = floor(invResolution * i + 0.00001f);
            float uf = invResolution * (i - resolution * vf + 0.5f) - 0.5f;
            vf = invResolution * (vf + 0.5f) - 0.5f;

            int u = (int)floor(uf * 32f / 4f);
            int v = (int)floor(vf * 32f / 4f);

            hashes[i] = hash.Eat(u).Eat(v);
        }
    }

    static int
        hashesID = Shader.PropertyToID("_Hashes"),
        configID = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh = default;

    [SerializeField]
    Material material = default;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-2f, 2f)]
    float verticalOffset = 1f;

    [SerializeField]
    int seed = 0;

    NativeArray<uint> hashes;

    ComputeBuffer hashesBuffer;

    MaterialPropertyBlock propertyBlock;

    private void OnEnable()
    {
        int length = resolution * resolution;
        hashes = new NativeArray<uint>(length, Allocator.Persistent);
        hashesBuffer = new ComputeBuffer(length, 4);

        new HashJob
        {
            hashes = hashes,
            resolution = resolution,
            invResolution = 1.0f / resolution,
            hash = SmallXXHash.Seed(seed)
        }.ScheduleParallel(hashes.Length, resolution, default).Complete();

        hashesBuffer.SetData(hashes);

        propertyBlock ??= new MaterialPropertyBlock();
        propertyBlock.SetBuffer(hashesID, hashesBuffer);
        propertyBlock.SetVector(configID, new Vector4(resolution, 1f / resolution, verticalOffset / resolution));
    }

    private void OnDisable()
    {
        hashes.Dispose();
        hashesBuffer.Release();
        hashesBuffer = null;
    }

    private void OnValidate()
    {
        if(hashesBuffer != null && enabled)
        {
            OnDisable();
            OnEnable();
        }
    }

    private void Update()
    {
        Graphics.DrawMeshInstancedProcedural(
            instanceMesh, 0, material, new Bounds(Vector3.zero, Vector3.one), hashes.Length, propertyBlock);
    }
}
