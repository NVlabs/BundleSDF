#include "common.h"
#include "SIFTImageManager.h"

SIFTImageManager::SIFTImageManager(unsigned int maxImages /*= 500*/, unsigned int maxKeyPointsPerImage /*= 4096*/)
{
	m_maxNumImages = maxImages;
	m_maxKeyPointsPerImage = maxKeyPointsPerImage;

	m_timer = NULL;
	m_currentImage = 0;
	alloc();
}

SIFTImageManager::~SIFTImageManager()
{
	free();
}

SIFTImageGPU& SIFTImageManager::getImageGPU(unsigned int imageIdx)
{
	assert(m_bFinalizedGPUImage);
	return m_SIFTImagesGPU[imageIdx];
}

const SIFTImageGPU& SIFTImageManager::getImageGPU(unsigned int imageIdx) const
{
	assert(m_bFinalizedGPUImage);
	return m_SIFTImagesGPU[imageIdx];
}

unsigned int SIFTImageManager::getNumImages() const
{
	return (unsigned int)m_SIFTImagesGPU.size();
}

unsigned int SIFTImageManager::getNumKeyPointsPerImage(unsigned int imageIdx) const
{
	return m_numKeyPointsPerImage[imageIdx];
}

SIFTImageGPU& SIFTImageManager::createSIFTImageGPU()
{
	assert(m_SIFTImagesGPU.size() == 0 || m_bFinalizedGPUImage);
	assert(m_SIFTImagesGPU.size() < m_maxNumImages);

	unsigned int imageIdx = (unsigned int)m_SIFTImagesGPU.size();
	m_SIFTImagesGPU.push_back(SIFTImageGPU());

	SIFTImageGPU& imageGPU = m_SIFTImagesGPU.back();

	//imageGPU.d_keyPointCounter = d_keyPointCounters + imageIdx;
	imageGPU.d_keyPoints = d_keyPoints + m_numKeyPoints;
	imageGPU.d_keyPointDescs = d_keyPointDescs + m_numKeyPoints;

	m_bFinalizedGPUImage = false;
	return imageGPU;
}

void SIFTImageManager::finalizeSIFTImageGPU(unsigned int numKeyPoints)
{
	assert(numKeyPoints <= m_maxKeyPointsPerImage);
	assert(!m_bFinalizedGPUImage);

	m_numKeyPointsPerImagePrefixSum.push_back(m_numKeyPoints);
	m_numKeyPoints += numKeyPoints;
	m_numKeyPointsPerImage.push_back(numKeyPoints);
	m_bFinalizedGPUImage = true;
	m_currentImage = (unsigned int)m_SIFTImagesGPU.size() - 1;

	assert(getNumImages() == m_numKeyPointsPerImage.size());
	assert(getNumImages() == m_numKeyPointsPerImagePrefixSum.size());
}

ImagePairMatch& SIFTImageManager::getImagePairMatch(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset)
{
	assert(prevImageIdx < getNumImages());
	assert(curImageIdx < getNumImages());
	keyPointOffset = make_uint2(m_numKeyPointsPerImagePrefixSum[prevImageIdx], m_numKeyPointsPerImagePrefixSum[curImageIdx]);
	return m_currImagePairMatches[prevImageIdx];
}




void SIFTImageManager::initializeMatching()
{
	for (unsigned int r = 0; r < m_maxNumImages; r++) {
		ImagePairMatch& imagePairMatch = m_currImagePairMatches[r];
		imagePairMatch.d_numMatches = d_currNumMatchesPerImagePair + r;
		imagePairMatch.d_distances = d_currMatchDistances + r * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
		imagePairMatch.d_keyPointIndices = d_currMatchKeyPointIndices + r * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	}
}


void findTrack(const std::vector< std::vector<std::pair<uint2, float3>> >& corrPerKey, std::vector<bool>& marker,
	std::vector<std::pair<uint2, float3>>& track, unsigned int curKey)
{
	for (unsigned int i = 0; i < corrPerKey[curKey].size(); i++) {
		const auto& c = corrPerKey[curKey][i];
		if (!marker[c.first.y]) { //not assigned to a track already
			track.push_back(c);
			marker[c.first.y] = true;
			findTrack(corrPerKey, marker, track, c.first.y);
		}
	}
}

#define MAX_TRACK_CORR_ERROR 0.03f
void SIFTImageManager::computeTracks(const std::vector<float4x4>& trajectory, const std::vector<EntryJ>& correspondences, const std::vector<uint2>& correspondenceKeyIndices,
	std::vector< std::vector<std::pair<uint2, float3>> >& tracks) const {
	tracks.clear();
	const unsigned int numImages = getNumImages();
	std::vector< std::vector<std::pair<uint2, float3>> > corrPerKey(m_numKeyPoints); //(image,keyIndex)
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid()) {
			const uint2& keyIndices = correspondenceKeyIndices[i];
			float err = length(trajectory[corr.imgIdx_i] * corr.pos_i - trajectory[corr.imgIdx_j] * corr.pos_j);
			if (err < MAX_TRACK_CORR_ERROR) {
				corrPerKey[keyIndices.x].push_back(std::make_pair(make_uint2(corr.imgIdx_j, keyIndices.y), corr.pos_j));
				corrPerKey[keyIndices.y].push_back(std::make_pair(make_uint2(corr.imgIdx_i, keyIndices.x), corr.pos_i));
			}
			else {
				corrPerKey[keyIndices.x].push_back(std::make_pair(make_uint2(corr.imgIdx_j, keyIndices.y), make_float3(-std::numeric_limits<float>::infinity())));
				corrPerKey[keyIndices.y].push_back(std::make_pair(make_uint2(corr.imgIdx_i, keyIndices.x), make_float3(-std::numeric_limits<float>::infinity())));
			}
		}
	}

	std::vector<bool> marker(m_numKeyPoints, false);
	for (unsigned int i = 0; i < numImages; i++) {
		for (unsigned int k = 0; k < m_numKeyPointsPerImage[i]; k++) {
			//std::vector<std::pair<uint2, float3>> track; track.reserve(numImages);
			//findTrack(corrPerKey, marker, track, m_numKeyPointsPerImagePrefixSum[i] + k);
			//if (!track.empty()) tracks.push_back(track);
			if (tracks.empty() || !tracks.back().empty()) tracks.push_back(std::vector<std::pair<uint2, float3>>()); //TODO how is this different?
			findTrack(corrPerKey, marker, tracks.back(), m_numKeyPointsPerImagePrefixSum[i] + k);
		}
	}
}

