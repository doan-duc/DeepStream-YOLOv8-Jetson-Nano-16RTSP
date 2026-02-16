/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 * Optimized for YOLOv8 (5x8400) - Missing Labels & Giant BBox Fix
 */

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include "utils.h"

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo b;

    // Kẹp tọa độ chặt chẽ để tránh BBox khổng lồ
    float x1 = std::max(0.0f, std::min((float)netW, bx1));
    float y1 = std::max(0.0f, std::min((float)netH, by1));
    float x2 = std::max(0.0f, std::min((float)netW, bx2));
    float y2 = std::max(0.0f, std::min((float)netH, by2));

    float width  = x2 - x1;
    float height = y2 - y1;

    // Lọc bỏ box có kích thước không hợp lệ hoặc quá lớn (nhiễu)
    if (width <= 2 || height <= 2 || width >= (netW - 2) || height >= (netH - 2)) return;

    b.left = x1;
    b.top = y1;
    b.width = width;
    b.height = height;
    b.detectionConfidence = maxProb;
    b.classId = maxIndex; // Gán classId = 0 để khớp với labels.txt

    binfo.push_back(b);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYoloV8(const float* output, const uint& numDetections, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold)
{
    std::vector<NvDsInferParseObjectInfo> binfo;
    for (uint i = 0; i < numDetections; ++i) {
        float maxProb = output[i + numDetections * 4];
        if (maxProb < preclusterThreshold[0]) continue;

        int maxIndex = 0; 

        // THÊM DÒNG NÀY: In ra màn hình Terminal để kiểm tra
        printf("DETECTED: Class %d | Conf: %.2f\n", maxIndex, maxProb);

        float cx = output[i];
        float cy = output[i + numDetections];
        float w  = output[i + numDetections * 2];
        float h  = output[i + numDetections * 3];

        float bx1 = std::max(0.0f, cx - w / 2);
        float by1 = std::max(0.0f, cy - h / 2);
        float bx2 = std::min((float)netW, cx + w / 2);
        float by2 = std::min((float)netH, cy + h / 2);

        if (bx2 > bx1 && by2 > by1) {
            addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }
    }
    return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    const uint numDetections = output.inferDims.d[1]; // 8400

    objectList = decodeTensorYoloV8((const float*)(output.buffer), numDetections,
        networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);

    return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
