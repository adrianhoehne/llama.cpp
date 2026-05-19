<script lang="ts">
	import { onMount } from 'svelte';
	import { ArrowLeft } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { APP_NAME, ROUTES } from '$lib/constants';
	import { MoeLayerPerfService } from '$lib/services';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import type {
		ApiMoeLayerPerfLayer,
		ApiMoeLayerPerfResponse,
		ApiMoeLayerPerfSummary
	} from '$lib/types/api';

	const GRAPH_WIDTH = 960;
	const GRAPH_HEIGHT = 120;
	const GRAPH_PADDING = 10;

	type CountMap = Map<number, number>;

	type ViewLayer = ApiMoeLayerPerfLayer & {
		hotByExpert: CountMap;
		coldByExpert: CountMap;
		activeByExpert: CountMap;
	};

	type GraphPoint = {
		x: number;
		y: number;
		ratio: number;
		layer: number;
	};

	type TimingMetric = {
		key: keyof ApiMoeLayerPerfSummary;
		label: string;
		unit: 'us' | 'count';
		description: string;
	};

	type TimingMetricGroup = {
		title: string;
		description?: string;
		metrics: TimingMetric[];
	};

	const TIMING_GROUPS_BEFORE: TimingMetricGroup[] = [
		{
			title: 'Summary',
			description: 'Top-level counters for the current perf window.',
			metrics: [
				{
					key: 'total_moe_time_per_call_us',
					label: 'Total MoE',
					unit: 'us',
					description: 'Average total MoE processing time per layer call.'
				},
				{
					key: 'layer_calls',
					label: 'Layer calls',
					unit: 'count',
					description: 'Total number of MoE layer calls included in the current perf window.'
				},
				{
					key: 'parallel_fallbacks',
					label: 'Fallbacks',
					unit: 'count',
					description: 'Total number of parallel scheduler fallbacks recorded in the current perf window.'
				}
			]
		},
		{
			title: '1. Routing / prep',
			description: 'Work before the hot and cold lanes can run.',
			metrics: [
				{
					key: 'routing_time_per_call_us',
					label: 'Routing',
					unit: 'us',
					description: 'Average time spent on gate routing and top-k preparation.'
				},
				{
					key: 'worklist_time_per_call_us',
					label: 'Worklist',
					unit: 'us',
					description: 'Average time spent building and compacting hot/cold expert worklists.'
				}
			]
		},
		{
			title: '2. Parallel region',
			description: 'Outer wall time of the hot/cold parallel scheduler region.',
			metrics: [
				{
					key: 'parallel_region_wall_time_per_call_us',
					label: 'Parallel wall',
					unit: 'us',
					description:
						'Average wall time of the hot/cold parallel scheduler region. This is not the sum of the lane internals.'
				}
			]
		}
	];

	const HOT_LANE_METRICS: TimingMetricGroup = {
		title: '2a. Hot lane',
		description: 'GPU hot-cache side of the parallel region.',
		metrics: [
			{
				key: 'parallel_hot_lane_wall_time_per_call_us',
				label: 'Hot lane',
				unit: 'us',
				description: 'Average wall time of the hot lane inside the parallel region.'
			},
			{
				key: 'hot_branch_time_per_call_us',
				label: 'Hot branch',
				unit: 'us',
				description:
					'Average measured time inside the hot-cache branch. This is an internal measurement and is not additive with lane wall time.'
			},
			{
				key: 'hot_gather_scatter_time_per_call_us',
				label: 'Hot gather',
				unit: 'us',
				description: 'Average gather/scatter overhead for hot-cache experts.'
			},
			{
				key: 'hot_expert_matmul_time_per_call_us',
				label: 'Hot matmul',
				unit: 'us',
				description: 'Average expert matrix multiplication time in the hot-cache branch.'
			},
			{
				key: 'parallel_hot_launches',
				label: 'Hot launches',
				unit: 'count',
				description: 'Total number of hot-lane launches recorded in the current perf window.'
			}
		]
	};

	const COLD_LANE_METRICS: TimingMetricGroup = {
		title: '2b. Cold lane',
		description: 'Cold expert side of the parallel region.',
		metrics: [
			{
				key: 'parallel_cold_lane_wall_time_per_call_us',
				label: 'Cold lane',
				unit: 'us',
				description: 'Average wall time of the cold lane inside the parallel region.'
			},
			{
				key: 'cold_branch_time_per_call_us',
				label: 'Cold branch',
				unit: 'us',
				description:
					'Average measured time inside the cold expert branch. This is an internal measurement and is not additive with lane wall time.'
			},
			{
				key: 'cold_gather_scatter_time_per_call_us',
				label: 'Cold gather',
				unit: 'us',
				description: 'Average gather/scatter overhead for cold experts.'
			},
			{
				key: 'cold_expert_matmul_time_per_call_us',
				label: 'Cold matmul',
				unit: 'us',
				description: 'Average expert matrix multiplication time in the cold branch.'
			},
			{
				key: 'parallel_cold_launches',
				label: 'Cold launches',
				unit: 'count',
				description: 'Total number of cold-lane launches recorded in the current perf window.'
			}
		]
	};

	const TIMING_GROUPS_AFTER: TimingMetricGroup[] = [
		{
			title: '3. Synchronization',
			description: 'What the scheduler gains or loses when the two lanes meet.',
			metrics: [
				{
					key: 'parallel_overlap_estimate_per_call_us',
					label: 'Overlap',
					unit: 'us',
					description: 'Estimated average time hidden by hot/cold parallel overlap.'
				},
				{
					key: 'parallel_join_wait_time_per_call_us',
					label: 'Join wait',
					unit: 'us',
					description: 'Average time spent waiting at the join point after one lane finishes earlier.'
				}
			]
		},
		{
			title: '4. Merge',
			description: 'Final output merge after hot and cold results are ready.',
			metrics: [
				{
					key: 'merge_time_per_call_us',
					label: 'Merge',
					unit: 'us',
					description:
						'Average time spent merging hot and cold branch outputs back into the layer output.'
				}
			]
		}
	];

	let perf = $state<ApiMoeLayerPerfResponse | null>(null);
	let previousPerf = $state<ApiMoeLayerPerfResponse | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);
	let lastUpdated = $state<Date | null>(null);
	let updateIntervalSeconds = $state(1);
	let mounted = $state(false);
	let lastRequestModel = $state<string | null>(null);

	let requestModel = $derived(isRouterMode() ? modelsStore.selectedModelName : null);
	let updateIntervalMs = $derived(Math.round(clamp(updateIntervalSeconds, 0.5, 3) * 1000));
	let visualLayers = $derived(toViewLayers(perf?.layers ?? [], previousPerf?.layers ?? []));
	let nExpert = $derived(resolveExpertCount(perf, visualLayers));
	let gridSide = $derived(Math.max(1, Math.ceil(Math.sqrt(nExpert))));
	let expertIds = $derived(
		Array.from({ length: gridSide * gridSide }, (_, index) => (index < nExpert ? index : -1))
	);
	let graphPoints = $derived(buildGraphPoints(visualLayers));
	let graphPolyline = $derived(graphPoints.map((point) => `${point.x},${point.y}`).join(' '));
	let averageHitRate = $derived(resolveAverageHitRate(perf, visualLayers));
	let activeExpertCount = $derived(
		visualLayers.reduce((sum, layer) => sum + layer.activeByExpert.size, 0)
	);

	function clamp(value: number, min: number, max: number): number {
		return Math.min(max, Math.max(min, value));
	}

	function countsToMap(counts?: Array<[number, number]>): CountMap {
		const result = new Map<number, number>();

		for (const pair of counts ?? []) {
			const [expert, count] = pair;
			if (Number.isFinite(expert) && Number.isFinite(count) && count > 0) {
				result.set(expert, count);
			}
		}

		return result;
	}

	function addCounts(target: CountMap, counts?: Array<[number, number]>) {
		for (const [expert, count] of counts ?? []) {
			if (Number.isFinite(expert) && Number.isFinite(count) && count > 0) {
				target.set(expert, (target.get(expert) ?? 0) + count);
			}
		}
	}

	function totalCountsForLayer(layer?: ApiMoeLayerPerfLayer): CountMap {
		const result = new Map<number, number>();
		if (!layer) {
			return result;
		}

		addCounts(result, layer.hot_experts);
		addCounts(result, layer.cold_experts);

		if (result.size === 0) {
			addCounts(result, layer.experts);
		}

		return result;
	}

	function activeDeltaForLayer(layer: ApiMoeLayerPerfLayer, previous?: ApiMoeLayerPerfLayer): CountMap {
		const current = totalCountsForLayer(layer);
		const before = totalCountsForLayer(previous);
		const result = new Map<number, number>();

		for (const [expert, count] of current) {
			const delta = count - (before.get(expert) ?? 0);
			if (delta > 0) {
				result.set(expert, delta);
			}
		}

		return result;
	}

	function toViewLayers(
		layers: ApiMoeLayerPerfLayer[],
		previousLayers: ApiMoeLayerPerfLayer[]
	): ViewLayer[] {
		const previousByLayer = new Map(previousLayers.map((layer) => [layer.layer, layer]));

		return layers
			.slice()
			.sort((a, b) => a.layer - b.layer)
			.map((layer) => ({
				...layer,
				hotByExpert: countsToMap(layer.hot_experts),
				coldByExpert: countsToMap(layer.cold_experts),
				activeByExpert: activeDeltaForLayer(layer, previousByLayer.get(layer.layer))
			}));
	}

	function resolveExpertCount(data: ApiMoeLayerPerfResponse | null, layers: ViewLayer[]): number {
		if (data?.n_expert && data.n_expert > 0) {
			return data.n_expert;
		}

		let maxExpert = -1;
		for (const layer of layers) {
			for (const expert of [...layer.hotByExpert.keys(), ...layer.coldByExpert.keys()]) {
				maxExpert = Math.max(maxExpert, expert);
			}
		}

		return Math.max(1, maxExpert + 1);
	}

	function layerHitRate(layer: ApiMoeLayerPerfLayer): number {
		if (typeof layer.hot_slot_ratio === 'number' && Number.isFinite(layer.hot_slot_ratio)) {
			return clamp(layer.hot_slot_ratio, 0, 1);
		}

		const hot = layer.hot_slots_total ?? 0;
		const cold = layer.cold_slots_total ?? 0;
		const total = hot + cold;

		return total > 0 ? hot / total : 0;
	}

	function resolveAverageHitRate(
		data: ApiMoeLayerPerfResponse | null,
		layers: ViewLayer[]
	): number {
		if (
			typeof data?.summary?.hot_slot_ratio === 'number' &&
			Number.isFinite(data.summary.hot_slot_ratio)
		) {
			return clamp(data.summary.hot_slot_ratio, 0, 1);
		}

		if (layers.length === 0) {
			return 0;
		}

		const total = layers.reduce((sum, layer) => sum + layerHitRate(layer), 0);
		return total / layers.length;
	}

	function buildGraphPoints(layers: ViewLayer[]): GraphPoint[] {
		const width = GRAPH_WIDTH - GRAPH_PADDING * 2;
		const height = GRAPH_HEIGHT - GRAPH_PADDING * 2;
		const lastIndex = Math.max(1, layers.length - 1);

		return layers.map((layer, index) => {
			const ratio = layerHitRate(layer);

			return {
				x: GRAPH_PADDING + (width * index) / lastIndex,
				y: GRAPH_PADDING + height * (1 - ratio),
				ratio,
				layer: layer.layer
			};
		});
	}

	function numberLocale(): string | string[] | undefined {
		if (typeof navigator === 'undefined') {
			return undefined;
		}

		return navigator.languages.length > 0 ? Array.from(navigator.languages) : navigator.language;
	}

	function formatDecimal(value: number, digits: number): string {
		return new Intl.NumberFormat(numberLocale(), {
			minimumFractionDigits: digits,
			maximumFractionDigits: digits
		}).format(value);
	}

	function formatInteger(value: number): string {
		return new Intl.NumberFormat(numberLocale(), {
			maximumFractionDigits: 0
		}).format(Math.round(value));
	}

	function formatPercent(value: number): string {
		return `${formatDecimal(value * 100, 1)}%`;
	}

	function metricValue(metric: TimingMetric): number | undefined {
		const value = perf?.summary?.[metric.key];
		return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
	}

	function formatMetricValue(metric: TimingMetric): string {
		const value = metricValue(metric);
		if (value === undefined) {
			return 'n/a';
		}

		if (metric.unit === 'count') {
			return formatInteger(value);
		}

		const digits = value >= 100 ? 1 : value >= 10 ? 2 : 3;
		return `${formatDecimal(value, digits)} us`;
	}

	function formatMetricShare(metric: TimingMetric): string | null {
		if (metric.unit !== 'us' || metric.key === 'total_moe_time_per_call_us') {
			return null;
		}

		const value = metricValue(metric);
		const total = perf?.summary?.total_moe_time_per_call_us;
		if (
			value === undefined ||
			typeof total !== 'number' ||
			!Number.isFinite(total) ||
			total <= 0
		) {
			return null;
		}

		return `${formatDecimal((value / total) * 100, 1)}%`;
	}

	function handleBack() {
		if (window.history.length > 1) {
			window.history.back();
			return;
		}

		window.location.href = ROUTES.START;
	}

	function expertClass(layer: ViewLayer, expert: number): string {
		const hot = layer.hotByExpert.get(expert) ?? 0;
		const cold = layer.coldByExpert.get(expert) ?? 0;
		const active = layer.activeByExpert.has(expert);
		const base =
			'aspect-square rounded-[2px] border border-background/30 transition-colors duration-150';
		const color = active
			? 'bg-yellow-400'
			: hot === 0 && cold === 0
				? 'bg-muted'
				: hot >= cold
					? 'bg-red-500'
					: 'bg-blue-500';

		return `${base} ${color}`;
	}

	function expertTitle(layer: ViewLayer, expert: number): string {
		const hot = layer.hotByExpert.get(expert) ?? 0;
		const cold = layer.coldByExpert.get(expert) ?? 0;
		const active = layer.activeByExpert.get(expert) ?? 0;

		return `Layer ${layer.layer}, expert ${expert}: hot ${hot}, cold ${cold}, active ${active}`;
	}

	function expertLegendClass(type: 'hot' | 'cold' | 'active' | 'idle'): string {
		const base = 'inline-block h-2.5 w-2.5 rounded-[2px]';

		if (type === 'hot') {
			return `${base} bg-red-500`;
		}
		if (type === 'cold') {
			return `${base} bg-blue-500`;
		}
		if (type === 'active') {
			return `${base} bg-yellow-400`;
		}

		return `${base} bg-muted`;
	}

	function handleIntervalInput(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const parsed = Number(input.value);
		const next = Number.isFinite(parsed) ? clamp(parsed, 0.5, 3) : 1;

		updateIntervalSeconds = Number(next.toFixed(1));
		input.value = String(updateIntervalSeconds);
	}

	async function refresh() {
		if (loading) {
			return;
		}

		loading = true;

		try {
			if (isRouterMode() && modelsStore.models.length === 0) {
				await modelsStore.fetch();
			}

			const model = requestModel;
			const nextPerf = await MoeLayerPerfService.get(model);

			previousPerf = model === lastRequestModel ? perf : null;
			lastRequestModel = model;
			perf = nextPerf;
			error = null;
			lastUpdated = new Date();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Failed to load MoE layer performance';
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		mounted = true;
		void refresh();
	});

	$effect(() => {
		if (!mounted) {
			return;
		}

		const interval = window.setInterval(() => {
			void refresh();
		}, updateIntervalMs);

		return () => window.clearInterval(interval);
	});
</script>

{#snippet timingMetricCard(metric: TimingMetric)}
	{@const share = formatMetricShare(metric)}
	<Tooltip.Root>
		<Tooltip.Trigger
			type="button"
			class="block w-full rounded-sm bg-muted-foreground/10 px-2 py-1.5 text-left transition-colors hover:bg-muted-foreground/15"
		>
			<span class="block truncate text-[11px] leading-4 text-muted-foreground">
				{metric.label}
			</span>
			<span class="flex items-baseline justify-between gap-1">
				<span class="truncate font-mono text-xs tabular-nums text-foreground">
					{formatMetricValue(metric)}
				</span>
				{#if share}
					<span class="text-[10px] tabular-nums text-muted-foreground">{share}</span>
				{/if}
			</span>
		</Tooltip.Trigger>
		<Tooltip.Content class="max-w-72">
			<p>{metric.description}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

{#snippet timingMetricGroup(group: TimingMetricGroup)}
	<section class="rounded-sm border border-border/60 bg-muted-foreground/5 p-2">
		<div>
			<h3 class="text-xs font-semibold text-foreground">{group.title}</h3>
			{#if group.description}
				<p class="mt-0.5 text-[11px] leading-4 text-muted-foreground">{group.description}</p>
			{/if}
		</div>

		<div class="mt-2 grid grid-cols-2 gap-2">
			{#each group.metrics as metric (metric.key)}
				{@render timingMetricCard(metric)}
			{/each}
		</div>
	</section>
{/snippet}

<svelte:head>
	<title>MoE Layer Perf · {APP_NAME}</title>
</svelte:head>

<div class="flex h-full min-h-0 flex-col bg-background">
	<header class="border-b bg-background/95 px-4 py-3 backdrop-blur-sm md:px-6">
		<div class="flex flex-wrap items-center justify-between gap-3">
			<div class="flex min-w-0 items-center gap-3">
				<Button variant="ghost" size="icon" onclick={handleBack}>
					<ArrowLeft class="h-4 w-4" />
					<span class="sr-only">Back</span>
				</Button>

				<div class="min-w-0">
					<h1 class="truncate text-base font-semibold">MoE layer performance</h1>
					<p class="truncate text-xs text-muted-foreground">
						{requestModel ?? 'active model'}
					</p>
				</div>
			</div>

			<div class="flex flex-wrap items-center gap-3">
				<div class="flex items-center gap-2 text-xs text-muted-foreground">
					<span>Update</span>
					<Input
						type="number"
						min="0.5"
						max="3"
						step="0.1"
						bind:value={updateIntervalSeconds}
						oninput={handleIntervalInput}
						class="h-8 w-20 text-right"
					/>
					<span>s</span>
				</div>
			</div>
		</div>
	</header>

	<main class="min-h-0 flex-1 overflow-auto px-4 py-4 md:px-6">
		<div class="mx-auto flex w-full max-w-7xl flex-col gap-4">
			<section class="rounded-md border bg-muted-foreground/5 p-4">
				<div class="space-y-4">
					<div class="min-w-0">
						<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
							<div>
								<h2 class="text-sm font-semibold">Hit rate</h2>
								<p class="text-xs text-muted-foreground">
									{formatPercent(averageHitRate)} average · {visualLayers.length} layers · {activeExpertCount}
									active experts since last update
								</p>
							</div>

							{#if lastUpdated}
								<p class="text-xs text-muted-foreground">{lastUpdated.toLocaleTimeString()}</p>
							{/if}
						</div>

						<div class="h-36 w-full overflow-hidden rounded-md bg-background/80">
							<svg
								viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
								class="h-full w-full text-muted-foreground"
								preserveAspectRatio="none"
								role="img"
								aria-label="MoE layer hit rate"
							>
								<line
									x1={GRAPH_PADDING}
									y1={GRAPH_HEIGHT - GRAPH_PADDING}
									x2={GRAPH_WIDTH - GRAPH_PADDING}
									y2={GRAPH_HEIGHT - GRAPH_PADDING}
									stroke="currentColor"
									stroke-opacity="0.25"
								/>

								{#each graphPoints as point}
									<line
										x1={point.x}
										y1={GRAPH_HEIGHT - GRAPH_PADDING}
										x2={point.x}
										y2={point.y}
										stroke="currentColor"
										stroke-opacity="0.25"
										stroke-width="3"
									/>
								{/each}

								{#if graphPolyline}
									<polyline
										points={graphPolyline}
										fill="none"
										stroke="var(--primary)"
										stroke-width="3"
										vector-effect="non-scaling-stroke"
									/>
								{/if}
							</svg>
						</div>

						<div class="mt-3 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
							<span class="inline-flex items-center gap-1">
								<span class={expertLegendClass('hot')}></span>
								Hot
							</span>
							<span class="inline-flex items-center gap-1">
								<span class={expertLegendClass('cold')}></span>
								Cold
							</span>
							<span class="inline-flex items-center gap-1">
								<span class={expertLegendClass('active')}></span>
								Active
							</span>
							<span class="inline-flex items-center gap-1">
								<span class={expertLegendClass('idle')}></span>
								No hits
							</span>
						</div>
					</div>

					<aside class="rounded-md border bg-background/70 p-3">
						<div class="mb-2">
							<h2 class="text-sm font-semibold">Timings</h2>
							<p class="text-xs text-muted-foreground">Ordered by MoE execution flow</p>
						</div>

						<div class="space-y-2">
							{#each TIMING_GROUPS_BEFORE as group (group.title)}
								{@render timingMetricGroup(group)}
							{/each}

							<div class="grid gap-2 sm:grid-cols-2">
								{@render timingMetricGroup(HOT_LANE_METRICS)}
								{@render timingMetricGroup(COLD_LANE_METRICS)}
							</div>

							{#each TIMING_GROUPS_AFTER as group (group.title)}
								{@render timingMetricGroup(group)}
							{/each}
						</div>
					</aside>
				</div>
			</section>

			{#if error}
				<div class="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
					{error}
				</div>
			{/if}

			{#if perf && !perf.enabled}
				<div class="rounded-md border bg-muted-foreground/5 p-4 text-sm text-muted-foreground">
					MoE layer performance is disabled.
				</div>
			{:else if visualLayers.length === 0}
				<div class="rounded-md border bg-muted-foreground/5 p-4 text-sm text-muted-foreground">
					No MoE layer data available.
				</div>
			{:else}
				<section class="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-4">
					{#each visualLayers as layer (layer.layer)}
						<article class="rounded-md border bg-muted-foreground/5 p-2">
							<div class="mb-2 flex items-center justify-between gap-2 text-xs">
								<span class="font-medium">Layer {layer.layer}</span>
								<span class="tabular-nums text-muted-foreground">{formatPercent(layerHitRate(layer))}</span>
							</div>

							<div
								class="grid aspect-square gap-px"
								style={`grid-template-columns: repeat(${gridSide}, minmax(0, 1fr));`}
							>
								{#each expertIds as expert, index (expert >= 0 ? expert : `blank-${index}`)}
									{#if expert >= 0}
										<div class={expertClass(layer, expert)} title={expertTitle(layer, expert)}></div>
									{:else}
										<div class="aspect-square rounded-[2px] bg-muted/40"></div>
									{/if}
								{/each}
							</div>
						</article>
					{/each}
				</section>
			{/if}
		</div>
	</main>
</div>
