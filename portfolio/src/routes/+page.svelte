<script lang="ts">
    import { onMount } from 'svelte';

    let activeSubheading: number = 0;
    let content: string = "";
    let mounted = false;

    // Subheading specific content
    const subheadingContent: { [key: number]: string } = {
        1: "This is the content for Subheading 1",
        2: "This is the content for Subheading 2",
        3: "This is the content for Subheading 3",
    };

    // Change content based on clicked subheading
    function changeContent(subheading: number) {
        activeSubheading = subheading;
        content = subheadingContent[subheading];
    }

    onMount(() => {
        setTimeout(() => {
            activeSubheading = 1;
            mounted = true;
        }, 100);
    });

    $: content = subheadingContent[activeSubheading] || "";
</script>

<style>
    body {
        margin: 0;
        padding: 0;
    }

    .container {
        position: relative;
        display: flex;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
    }

    .left {
        position: relative;
        z-index: 3;
        background-color: #D3D9D4;
        width: 20%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-left: 20px;
    }

    .center,
    .right {
        /* position: relative; */
    }

    .center {
        z-index: 2;
        background-color: #748D92;
        width: 40%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        transform: translateX(-100%);
        transition: transform 1s ease-in-out;
    }

    .right {
        z-index: 1;
        background-color: #124E66;
        width: 40%;
        transform: translateX(-200%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        transition: transform 2s ease-in-out;
    }

    .center.active,
    .right.active {
        transform: translateX(0);
    }

    h1 {
        margin: 0 0 20px;
        font-size: 24px;
    }

    .subheadings {
        cursor: pointer;
        font-size: 18px;
        margin: 10px 0;
    }
</style>

<div class="container">
    <!-- Left column -->
    <div class="left">
        <h1>HEADING 1</h1>
        <div>
            <div class="subheadings" on:click={() => changeContent(1)}>SUBHEADING 1</div>
            <div class="subheadings" on:click={() => changeContent(2)}>SUBHEADING 2</div>
            <div class="subheadings" on:click={() => changeContent(3)}>SUBHEADING 3</div>
        </div>
    </div>

    <!-- Center column -->
    <div class="center" class:active={mounted && activeSubheading !== 0}>
        <h1>HEADING 2</h1>
        <p>{content}</p>
    </div>

    <!-- Right column -->
    <div class="right" class:active={mounted && activeSubheading !== 0}>
        <h1>HEADING 3</h1>
    </div>
</div>
