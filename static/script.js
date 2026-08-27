// script.js - Gradio 6 兼容版（基于 AUTOMATIC1111 的脚本，适配动态渲染）
(function() {
    // 确保 waitForElm 存在（如果 viewer.js 未加载，则自行定义）
    if (typeof waitForElm === 'undefined') {
        window.waitForElm = function(selector) {
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
        };
    }

    // Gradio 6 中 gradio-app 元素不再使用 shadowRoot，直接返回 document 或根容器
    function gradioApp() {
        // 优先返回 gradio-app 元素（如果有），否则返回 document
        const gradioAppElem = document.querySelector('gradio-app');
        if (gradioAppElem) {
            // 模拟 getElementById 直接指向 document（简化）
            gradioAppElem.getElementById = function(id) {
                return document.getElementById(id);
            };
            return gradioAppElem;
        }
        // 如果没有 gradio-app，则返回 document（兼容旧方式）
        return document;
    }

    /**
     * Get the currently selected top-level UI tab button.
     * Gradio 6 中 tabs 结构可能变化，保留原选择器并加入容错。
     */
    function get_uiCurrentTab() {
        const tabs = gradioApp().querySelector('#tabs > .tab-nav > button.selected');
        return tabs || null;
    }

    /**
     * Get the first currently visible top-level UI tab content.
     */
    function get_uiCurrentTabContent() {
        const tabContent = gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
        return tabContent || null;
    }

    // 回调队列
    let uiUpdateCallbacks = [];
    let uiAfterUpdateCallbacks = [];
    let uiLoadedCallbacks = [];
    let uiTabChangeCallbacks = [];
    let optionsChangedCallbacks = [];
    let uiAfterUpdateTimeout = null;
    let uiCurrentTab = null;
    let executedOnLoaded = false;

    function executeCallbacks(queue, arg) {
        for (const callback of queue) {
            try {
                callback(arg);
            } catch (e) {
                console.error("error running callback", callback, ":", e);
            }
        }
    }

    function scheduleAfterUiUpdateCallbacks() {
        clearTimeout(uiAfterUpdateTimeout);
        uiAfterUpdateTimeout = setTimeout(function() {
            executeCallbacks(uiAfterUpdateCallbacks);
        }, 200);
    }

    // 公开的注册函数
    window.onUiUpdate = function(callback) {
        uiUpdateCallbacks.push(callback);
    };
    window.onAfterUiUpdate = function(callback) {
        uiAfterUpdateCallbacks.push(callback);
    };
    window.onUiLoaded = function(callback) {
        uiLoadedCallbacks.push(callback);
    };
    window.onUiTabChange = function(callback) {
        uiTabChangeCallbacks.push(callback);
    };
    window.onOptionsChanged = function(callback) {
        optionsChangedCallbacks.push(callback);
    };

    // 初始化 UI 观察器（等待 Gradio 根容器出现）
    async function initUiObserver() {
        await waitForElm('.gradio-container');

        // 如果尚未触发 onUiLoaded，检查生成按钮是否存在
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        const mutationObserver = new MutationObserver(function(mutations) {
            // 持续检查生成按钮是否出现（用于 onUiLoaded）
            if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
                executedOnLoaded = true;
                executeCallbacks(uiLoadedCallbacks);
            }

            executeCallbacks(uiUpdateCallbacks, mutations);
            scheduleAfterUiUpdateCallbacks();

            const newTab = get_uiCurrentTab();
            if (newTab && (newTab !== uiCurrentTab)) {
                uiCurrentTab = newTab;
                executeCallbacks(uiTabChangeCallbacks);
            }
        });

        mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
    }

    // 启动观察器
    initUiObserver().catch(err => console.warn("UI observer init failed:", err));

    // Ctrl+Enter 快捷键生成或停止
    document.addEventListener('keydown', function(e) {
        const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
        const isEnterKey = (e.key == "Enter" || e.keyCode == 13);
        if (isModifierKey && isEnterKey) {
            const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
            if (generateButton) {
                generateButton.click();
                e.preventDefault();
                return;
            }
            const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]');
            if (stopButton) {
                stopButton.click();
                e.preventDefault();
                return;
            }
        }
    });

    // 样式预览悬浮层（等待样式选择区域出现后再初始化）
    function initStylePreviewOverlay() {
        waitForElm('.style_selections').then(() => {
            // 获取 samples-path meta 标签，如果不存在则跳过
            const samplesPathMeta = document.querySelector("meta[name='samples-path']");
            if (!samplesPathMeta) {
                console.warn("Style preview: meta[name='samples-path'] not found");
                return;
            }
            const samplesPath = samplesPathMeta.getAttribute("content");
            let overlayVisible = false;
            const overlay = document.createElement('div');
            const tooltip = document.createElement('div');
            tooltip.className = 'preview-tooltip';
            overlay.appendChild(tooltip);
            overlay.id = 'stylePreviewOverlay';
            document.body.appendChild(overlay);

            let activeLabel = null;
            function onMouseLeave() {
                if (activeLabel) {
                    activeLabel.removeEventListener("mouseout", onMouseLeave);
                    activeLabel = null;
                }
                overlayVisible = false;
                overlay.style.opacity = "0";
                overlay.style.backgroundImage = "";
            }

            document.addEventListener('mouseover', function (e) {
                const label = e.target.closest('.style_selections label');
                if (!label) return;
                if (activeLabel === label) return;
                if (activeLabel) {
                    activeLabel.removeEventListener("mouseout", onMouseLeave);
                }
                activeLabel = label;
                label.addEventListener("mouseout", onMouseLeave);
                overlayVisible = true;
                overlay.style.opacity = "1";
                const originalText = label.querySelector("span")?.getAttribute("data-original-text");
                const name = originalText || label.querySelector("span")?.textContent || "";
                if (!name) return;
                // 构建图片 URL，将空格替换为下划线
                const imgName = name.toLowerCase().replaceAll(" ", "_");
                let imgUrl = samplesPath.replace("fooocus_v2", imgName);
                // 处理 Windows 路径反斜杠
                imgUrl = imgUrl.replaceAll("\\", "\\\\");
                overlay.style.backgroundImage = `url("${imgUrl}")`;
                tooltip.textContent = name;
            });

            document.addEventListener('mousemove', function (e) {
                if (!overlayVisible) return;
                overlay.style.left = `${e.clientX}px`;
                overlay.style.top = `${e.clientY}px`;
                overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
            });
        }).catch(err => console.warn("Style preview init failed:", err));
    }

    initStylePreviewOverlay();

    // 工具函数
    window.uiElementIsVisible = function(el) {
        if (el === document) return true;
        const computedStyle = getComputedStyle(el);
        const isVisible = computedStyle.display !== 'none';
        if (!isVisible) return false;
        return window.uiElementIsVisible(el.parentNode);
    };

    window.uiElementInSight = function(el) {
        const clRect = el.getBoundingClientRect();
        const windowHeight = window.innerHeight;
        const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;
        return isOnScreen;
    };

    window.playNotification = function() {
        const audio = gradioApp().querySelector('#audio_notification audio');
        if (audio) audio.play().catch(e => console.warn("Notification play failed:", e));
    };

    window.set_theme = function(theme) {
        let gradioURL = window.location.href;
        if (!gradioURL.includes('?__theme=')) {
            window.location.replace(gradioURL + '?__theme=' + theme);
        }
    };

    // 导出部分函数到全局（保持 API 兼容）
    window.gradioApp = gradioApp;
    window.get_uiCurrentTab = get_uiCurrentTab;
    window.get_uiCurrentTabContent = get_uiCurrentTabContent;
})();