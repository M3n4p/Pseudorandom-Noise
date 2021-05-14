using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

using static HashFunctions;
using static Domain;
using static Unity.Mathematics.math;

public class HashVisualization : MonoBehaviour
{
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast, CompileSynchronously = true)]
    struct HashJob : IJobFor
    {
        [ReadOnly]
        public NativeArray<float3> positions;

        [WriteOnly]
        public NativeArray<uint> hashes;

        public SmallXXHash hash;

        public float3x4 domainTRS;

        public void Execute(int i)
        {
            float3 p = mul(domainTRS, float4(positions[i], 1f));

            int u = (int)floor(p.x);
            int v = (int)floor(p.y);
            int w = (int)floor(p.z);

            hashes[i] = hash.Eat(u).Eat(v).Eat(w);
        }
    }

    static int
        hashesID = Shader.PropertyToID("_Hashes"),
        positionsID = Shader.PropertyToID("_Positions"),
        normalsID = Shader.PropertyToID("_Normals"),
        configID = Shader.PropertyToID("_Config");

    [SerializeField]
    Mesh instanceMesh = default;

    [SerializeField]
    Material material = default;

    [SerializeField, Range(1, 512)]
    int resolution = 16;

    [SerializeField, Range(-0.5f, 0.5f)]
    float displacement = 0.1f;

    [SerializeField]
    int seed = 0;

    [SerializeField]
    SpaceTRS domain = new SpaceTRS
    {
        scale = 8f
    };

    NativeArray<uint> hashes;
    NativeArray<float3> positions, normals;

    ComputeBuffer hashesBuffer, positionsBuffer, normalsBuffer;

    MaterialPropertyBlock propertyBlock;

    bool isDirty;

    private void OnEnable()
    {
        isDirty = true;
        int length = resolution * resolution;
        hashes = new NativeArray<uint>(length, Allocator.Persistent);
        positions = new NativeArray<float3>(length, Allocator.Persistent);
        normals = new NativeArray<float3>(length, Allocator.Persistent);
        hashesBuffer = new ComputeBuffer(length, 4);
        positionsBuffer = new ComputeBuffer(length, 3 * 4);
        normalsBuffer = new ComputeBuffer(length, 3 * 4);

        propertyBlock ??= new MaterialPropertyBlock();
        propertyBlock.SetBuffer(hashesID, hashesBuffer);
        propertyBlock.SetBuffer(positionsID, positionsBuffer);
        propertyBlock.SetBuffer(normalsID, normalsBuffer);
        propertyBlock.SetVector(configID, new Vector4(resolution, 1f / resolution, displacement));
    }

    private void OnDisable()
    {
        hashes.Dispose();
        positions.Dispose();
        hashesBuffer.Release();
        positionsBuffer.Release();
        normalsBuffer.Release();
        hashesBuffer = null;
        positionsBuffer = null;
        normalsBuffer = null;
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
        if(isDirty || transform.hasChanged)
        {
            isDirty = false;
            transform.hasChanged = false;

            JobHandle handle = Shapes.Job.ScheduleParallel(positions, normals, resolution, transform.localToWorldMatrix, default);

            new HashJob
            {
                positions = positions,
                hashes = hashes,
                hash = SmallXXHash.Seed(seed),
                domainTRS = domain.Matrix
            }.ScheduleParallel(hashes.Length, resolution, handle).Complete();

            hashesBuffer.SetData(hashes);
            positionsBuffer.SetData(positions);
            normalsBuffer.SetData(normals);
        }

        Graphics.DrawMeshInstancedProcedural(
            instanceMesh, 0, material, new Bounds(Vector3.zero, Vector3.one), hashes.Length, propertyBlock);
    }
}
