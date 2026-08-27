// viewer.js - Gradio 6 兼容版，使用 waitForElm 等待元素就绪

window.main_viewer_height = 512;

/**
 * 等待指定的 DOM 元素出现
 * @param {string} selector - CSS 选择器
 * @returns {Promise<Element>}
 */
function waitForElm(selector) {
    return new Promise(resolve => {
        const existing = document.querySelector(selector);
        if (existing) {
            resolve(existing);
            return;
        }
        const observer = new MutationObserver(() => {
            const el = document.querySelector(selector);
            if (el) {
                resolve(el);
                observer.disconnect();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    });
}

/**
 * 刷新网格布局（依赖 #final_gallery 和 .grid-container）
 */
function refresh_grid() {
    waitForElm('#final_gallery').then(final_gallery => {
        const gridContainer = final_gallery.querySelector('.grid-container');
        if (gridContainer) {
            const rect = final_gallery.getBoundingClientRect();
            let cols = Math.ceil((rect.width - 16.0) / rect.height);
            if (cols < 2) cols = 2;
            gridContainer.style.setProperty('--grid-cols', cols);
        }
    }).catch(console.warn);
}

function refresh_grid_delayed() {
    refresh_grid();
    setTimeout(refresh_grid, 100);
    setTimeout(refresh_grid, 500);
    setTimeout(refresh_grid, 1000);
}

function resized() {
    let windowHeight = window.innerHeight - 260;
    if (windowHeight > 745) windowHeight = 745;
    window.main_viewer_height = windowHeight;

    // 等待所有 .main_view 元素出现后调整高度
    waitForElm('.main_view').then(() => {
        const elements = document.querySelectorAll('.main_view');
        for (let i = 0; i < elements.length; i++) {
            elements[i].style.height = windowHeight + 'px';
        }
        refresh_grid();
    }).catch(console.warn);
}

function viewer_to_top(delay = 100) {
    setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), delay);
}

function viewer_to_bottom(delay = 100) {
    // 等待 positive_prompt 元素出现
    waitForElm('#positive_prompt').then(element => {
        let yPos = window.main_viewer_height;
        if (element) {
            yPos = element.getBoundingClientRect().top + window.scrollY;
        }
        setTimeout(() => window.scrollTo({ top: yPos - 8, behavior: 'smooth' }), delay);
    }).catch(() => {
        // 如果找不到 positive_prompt，回退到原逻辑（使用 main_viewer_height）
        setTimeout(() => window.scrollTo({ top: window.main_viewer_height - 8, behavior: 'smooth' }), delay);
    });
}

function on_style_selection_blur() {
    waitForElm('#gradio_receiver_style_selections textarea').then(target => {
        target.value = "on_style_selection_blur " + Math.random();
        const e = new Event("input", { bubbles: true });
        Object.defineProperty(e, "target", { value: target });
        target.dispatchEvent(e);
    }).catch(console.warn);
}

/**
 * 初始化所有需要等待元素的逻辑
 */
function initViewer() {
    // 1. 调整尺寸（初始和窗口大小变化时）
    resized();
    window.addEventListener('resize', () => resized());

    // 2. 处理 .aspect_ratios span 中的 HTML 实体
    waitForElm('.aspect_ratios').then(() => {
        const spans = document.querySelectorAll('.aspect_ratios span');
        spans.forEach(span => {
            span.innerHTML = span.innerHTML.replace(/&lt;/g, '<').replace(/&gt;/g, '>');
        });
    }).catch(console.warn);

    // 3. 为 style_selections 添加焦点离开事件
    waitForElm('.style_selections').then(styleDiv => {
        styleDiv.addEventListener('focusout', function (event) {
            setTimeout(() => {
                if (!this.contains(document.activeElement)) {
                    on_style_selection_blur();
                }
            }, 200);
        });
    }).catch(console.warn);

    // 4. 调整 lora_weight 的滑块样式
    waitForElm('.lora_weight input[type="range"]').then(() => {
        const inputs = document.querySelectorAll('.lora_weight input[type="range"]');
        inputs.forEach(input => {
            input.style.marginTop = '12px';
        });
    }).catch(console.warn);
}

// 启动初始化：等待 Gradio 的根容器出现后再执行
waitForElm('.gradio-container').then(initViewer).catch(() => {
    // 如果 gradio-container 不存在（极少数情况），降级使用 DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initViewer);
    } else {
        initViewer();
    }
});