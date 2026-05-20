<script lang="ts">
	import { Activity, Square, SkipForward } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		ChatFormActionsAdd,
		ChatFormActionModels,
		ChatFormActionRecord,
		ChatFormActionSubmit,
		ChatFormReasoningToggle
	} from '$lib/components/app';
	import { FileTypeCategory } from '$lib/enums';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { getFileTypeCategory } from '$lib/utils';
	import { goto } from '$app/navigation';
	import { ROUTES } from '$lib/constants/routes';
	import { ChatService, MoeLayerPerfService } from '$lib/services';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import type { ApiMoeLayerPerfMode } from '$lib/types/api';

	interface Props {
		canSend?: boolean;
		canSubmit?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isReasoning?: boolean;
		isRecording?: boolean;
		moePerfModel?: string | null;
		showAddButton?: boolean;
		showModelSelector?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		canSend = false,
		canSubmit = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isReasoning = false,
		isRecording = false,
		moePerfModel = null,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick
	}: Props = $props();

	let currentConfig = $derived(config());

	let hasMcpPromptsSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasPromptsCapability(perChatOverrides);
	});

	let hasMcpResourcesSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasResourcesCapability(perChatOverrides);
	});

	let hasAudioModality = $state(false);
	let hasVideoModality = $state(false);
	let hasVisionModality = $state(false);
	let hasModelSelected = $state(false);
	let isSelectedModelInCache = $state(true);
	let submitTooltip = $state('');

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !canSubmit && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let selectorModelRef: ChatFormActionModels | undefined = $state(undefined);
	let moePerfMode = $state<ApiMoeLayerPerfMode>('full');
	let moePerfModeLoading = $state(false);
	let moePerfModeError = $state<string | null>(null);
	let lastMoePerfTarget = $state<string | null>(null);

	let isRouter = $derived(isRouterMode());
	let moePerfTargetModel = $derived(isRouter ? moePerfModel : null);
	let moePerfTargetKey = $derived(`${isRouter ? 'router' : 'model'}:${moePerfTargetModel ?? ''}`);

	export function openModelSelector() {
		selectorModelRef?.open();
	}

	// the streaming assistant message carries both the completion id and the model that
	// produced it, targeting reasoning control from the same source keeps them consistent
	let activeMessage = $derived(
		conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1]
	);

	function modeFromResponse(data: { enabled: boolean; mode?: ApiMoeLayerPerfMode }): ApiMoeLayerPerfMode {
		return data.mode ?? (data.enabled ? 'full' : 'off');
	}

	async function refreshMoePerfMode() {
		if (isRouter && !moePerfTargetModel) {
			return;
		}

		moePerfModeLoading = true;
		try {
			const data = await MoeLayerPerfService.get(moePerfTargetModel);
			moePerfMode = modeFromResponse(data);
			moePerfModeError = null;
		} catch (cause) {
			moePerfModeError =
				cause instanceof Error ? cause.message : 'Failed to load MoE perf mode';
		} finally {
			moePerfModeLoading = false;
		}
	}

	async function handleMoePerfModeChange(event: Event) {
		const select = event.currentTarget as HTMLSelectElement;
		const nextMode = select.value as ApiMoeLayerPerfMode;
		const previousMode = moePerfMode;

		moePerfMode = nextMode;
		moePerfModeLoading = true;

		try {
			const data = await MoeLayerPerfService.setMode(nextMode, moePerfTargetModel);
			moePerfMode = modeFromResponse(data);
			moePerfModeError = null;
		} catch (cause) {
			moePerfMode = previousMode;
			select.value = previousMode;
			moePerfModeError =
				cause instanceof Error ? cause.message : 'Failed to update MoE perf mode';
		} finally {
			moePerfModeLoading = false;
		}
	}

	$effect(() => {
		if (lastMoePerfTarget === moePerfTargetKey) {
			return;
		}

		lastMoePerfTarget = moePerfTargetKey;
		void refreshMoePerfMode();
	});
</script>

<div
	class="flex w-full items-center gap-3 {className} {showAddButton ? '' : 'justify-end'}"
	style="container-type: inline-size"
>
	{#if showAddButton}
		<div class="mr-auto flex items-center gap-3">
			<ChatFormActionsAdd
				{disabled}
				{hasAudioModality}
				{hasVideoModality}
				{hasVisionModality}
				{hasMcpPromptsSupport}
				{hasMcpResourcesSupport}
				{onFileUpload}
				{onSystemPromptClick}
				{onMcpPromptClick}
				{onMcpResourcesClick}
				onMcpSettingsClick={() => goto(ROUTES.MCP_SERVERS)}
			/>

			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						href={ROUTES.MOE_LAYER_PERF}
						variant="ghost"
						size="icon"
						class="h-8 w-8 rounded-full text-muted-foreground hover:text-foreground"
					>
						<Activity class="h-4 w-4" />
						<span class="sr-only">MoE layer performance</span>
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>MoE layer performance</p>
				</Tooltip.Content>
			</Tooltip.Root>

			<select
				value={moePerfMode}
				disabled={disabled || moePerfModeLoading || (isRouter && !moePerfTargetModel)}
				onchange={handleMoePerfModeChange}
				class="h-8 rounded-md border border-input bg-background px-2 text-xs text-foreground shadow-sm outline-none transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
				aria-label="MoE performance mode"
				title={moePerfModeError ?? 'MoE performance counters'}
			>
				<option value="full">Full</option>
				<option value="update">Update</option>
				<option value="off">Off</option>
			</select>
		</div>
	{/if}

	<div class="flex items-center gap-2">
		<ChatFormReasoningToggle />

		{#if showModelSelector}
			<ChatFormActionModels
				{disabled}
				bind:this={selectorModelRef}
				bind:hasAudioModality
				bind:hasVideoModality
				bind:hasVisionModality
				bind:hasModelSelected
				bind:isSelectedModelInCache
				bind:submitTooltip
				forceForegroundText
				useGlobalSelection
			/>
		{/if}
	</div>

	{#if isReasoning}
		<Button
			type="button"
			variant="secondary"
			onclick={() =>
				ChatService.stopReasoning(activeMessage?.completionId ?? '', activeMessage?.model)}
			class="group h-8 w-8 rounded-full p-0"
			title="Skip reasoning"
		>
			<span class="sr-only">Skip reasoning</span>

			<SkipForward class="h-4 w-4 stroke-muted-foreground group-hover:stroke-foreground" />
		</Button>
	{/if}

	{#if isLoading && !canSubmit}
		<Button
			type="button"
			variant="secondary"
			onclick={onStop}
			class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && (showModelSelector ? hasModelSelected && isSelectedModelInCache : true)}
			{disabled}
			tooltipLabel={submitTooltip}
			showErrorState={showModelSelector && hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>
