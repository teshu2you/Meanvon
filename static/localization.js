// localization.js - Gradio 6 兼容版（等待元素就绪后执行翻译）

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

    // Gradio 6 中 gradioApp 直接返回 document
    function gradioApp() {
        return document;
    }

    const re_num = /^[.\d]+$/;
    let original_lines = {};
    let translated_lines = {};

    function hasLocalization() {
        return window.localization && Object.keys(window.localization).length > 0;
    }

    function textNodesUnder(el) {
        let n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
        while ((n = walk.nextNode())) a.push(n);
        return a;
    }

    function canBeTranslated(node, text) {
        if (!text) return false;
        if (!node.parentElement) return false;
        const parentType = node.parentElement.nodeName;
        if (parentType === 'SCRIPT' || parentType === 'STYLE' || parentType === 'TEXTAREA') return false;
        if (re_num.test(text)) return false;
        return true;
    }

    function getTranslation(text) {
        if (!text) return undefined;
        if (translated_lines[text] === undefined) {
            original_lines[text] = 1;
        }
        const tl = window.localization?.[text];
        if (tl !== undefined) {
            translated_lines[tl] = 1;
        }
        return tl;
    }

    function processTextNode(node) {
        const text = node.textContent.trim();
        if (!canBeTranslated(node, text)) return;
        const tl = getTranslation(text);
        if (tl !== undefined) {
            node.textContent = tl;
            if (text && node.parentElement) {
                node.parentElement.setAttribute("data-original-text", text);
            }
        }
    }

    function processNode(node) {
        if (node.nodeType === 3) {
            processTextNode(node);
            return;
        }
        if (node.title) {
            const tl = getTranslation(node.title);
            if (tl !== undefined) node.title = tl;
        }
        if (node.placeholder) {
            const tl = getTranslation(node.placeholder);
            if (tl !== undefined) node.placeholder = tl;
        }
        textNodesUnder(node).forEach(processTextNode);
    }

    // 刷新样式选择区域的本地化（特定函数保留）
    window.refresh_style_localization = function() {
        const styleSel = document.querySelector('.style_selections');
        if (styleSel) processNode(styleSel);
    };

    function localizeWholePage() {
        processNode(gradioApp());

        // 处理通过 gradio_config 定义的组件属性
        if (window.gradio_config && window.gradio_config.components) {
            for (const comp of window.gradio_config.components) {
                const elemId = comp.props?.elem_id ? comp.props.elem_id : "component-" + comp.id;
                const e = document.getElementById(elemId);
                if (!e) continue;

                if (comp.props?.webui_tooltip) {
                    const tl = e ? getTranslation(e.title) : undefined;
                    if (tl !== undefined) e.title = tl;
                }
                if (comp.props?.placeholder) {
                    const textbox = e.querySelector('[placeholder]');
                    if (textbox) {
                        const tl = getTranslation(textbox.placeholder);
                        if (tl !== undefined) textbox.placeholder = tl;
                    }
                }
            }
        }
    }

    // 启用 RTL 样式（如果语言从右向左）
    function enableRTLIfNeeded() {
        if (!window.localization?.rtl) return;
        const observer = new MutationObserver((mutations, obs) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.tagName === 'STYLE') {
                        obs.disconnect();
                        for (const rule of node.sheet?.rules || []) {
                            if (Array.from(rule.media || []).includes('rtl')) {
                                rule.media.appendMedium('all');
                            }
                        }
                        return;
                    }
                }
            }
        });
        observer.observe(document.head, { childList: true });
    }

    // 初始化翻译：等待页面根容器出现，然后执行全文翻译并设置动态监听
    async function initLocalization() {
        if (!hasLocalization()) return;

        // 等待 gradio 的根容器出现
        await waitForElm('.gradio-container');

        // 首次翻译整个页面
        localizeWholePage();

        // 使用 MutationObserver 监听动态添加的节点并自动翻译
        const observer = new MutationObserver(mutations => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === 1) { // 元素节点
                        processNode(node);
                    } else if (node.nodeType === 3) { // 文本节点
                        processTextNode(node);
                    }
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // 启用 RTL（如果需要）
        enableRTLIfNeeded();
    }

    initLocalization().catch(err => console.warn("Localization init failed:", err));
})();