import { apiFetch, apiFetchWithParams } from '$lib/utils';
import type { ApiMoeLayerPerfMode, ApiMoeLayerPerfResponse } from '$lib/types/api';

const MOE_LAYER_PERF_ENDPOINT = '/moe-layer-perf';

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
}
