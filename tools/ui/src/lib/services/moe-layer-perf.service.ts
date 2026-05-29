import { apiFetch, apiFetchWithParams } from '$lib/utils';
import type {
	ApiMoeHotCacheApplyResponse,
	ApiMoeLayerPerfMode,
	ApiMoeLayerPerfResponse
} from '$lib/types/api';

const MOE_LAYER_PERF_ENDPOINT = '/moe-layer-perf';
const MOE_HOT_CACHE_ENDPOINT = '/moe-hot-cache';

export class MoeLayerPerfService {
	static async get(model?: string | null): Promise<ApiMoeLayerPerfResponse> {
		if (model) {
			return apiFetchWithParams<ApiMoeLayerPerfResponse>(MOE_LAYER_PERF_ENDPOINT, {
				model,
				autoload: 'false'
			});
		}

		return apiFetch<ApiMoeLayerPerfResponse>(MOE_LAYER_PERF_ENDPOINT);
	}

	static async setMode(
		mode: ApiMoeLayerPerfMode,
		model?: string | null
	): Promise<ApiMoeLayerPerfResponse> {
		const options = {
			method: 'POST',
			body: JSON.stringify({ mode })
		};

		if (model) {
			return apiFetchWithParams<ApiMoeLayerPerfResponse>(
				MOE_LAYER_PERF_ENDPOINT,
				{
					model,
					autoload: 'false'
				},
				options
			);
		}

		return apiFetch<ApiMoeLayerPerfResponse>(MOE_LAYER_PERF_ENDPOINT, options);
	}

	static async applyHotCache(
		perf: ApiMoeLayerPerfResponse,
		model?: string | null,
		saveToDisk = false
	): Promise<ApiMoeHotCacheApplyResponse> {
		const options = {
			method: 'POST',
			body: JSON.stringify({
				...perf,
				save_to_disk: saveToDisk
			})
		};

		if (model) {
			return apiFetchWithParams<ApiMoeHotCacheApplyResponse>(
				MOE_HOT_CACHE_ENDPOINT,
				{
					model,
					autoload: 'false'
				},
				options
			);
		}

		return apiFetch<ApiMoeHotCacheApplyResponse>(MOE_HOT_CACHE_ENDPOINT, options);
	}
}
