// context_menu.js - Gradio 6 兼容版（使用 waitForElm）
(function() {
    // 确保 waitForElm 存在（如果 viewer.js 未加载，则自己定义）
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

    // Gradio 6 中 gradioApp() 不再是必需的，直接使用 document
    function gradioApp() {
        return document;
    }

    let eventListenerApplied = false;
    let menuSpecs = new Map();

    const uid = function() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };

    function showContextMenu(event, element, menuEntries) {
        let posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
        let posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;

        let oldMenu = gradioApp().querySelector('#context-menu');
        if (oldMenu) {
            oldMenu.remove();
        }

        let baseStyle = window.getComputedStyle(gradioApp().querySelector('button.selected'));

        const contextMenu = document.createElement('nav');
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = posy + 'px';
        contextMenu.style.left = posx + 'px';

        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        menuEntries.forEach(function(entry) {
            let contextMenuEntry = document.createElement('a');
            contextMenuEntry.innerHTML = entry['name'];
            contextMenuEntry.addEventListener("click", function() {
                entry['func']();
            });
            contextMenuList.append(contextMenuEntry);
        });

        gradioApp().appendChild(contextMenu);

        let menuWidth = contextMenu.offsetWidth + 4;
        let menuHeight = contextMenu.offsetHeight + 4;

        let windowWidth = window.innerWidth;
        let windowHeight = window.innerHeight;

        if ((windowWidth - posx) < menuWidth) {
            contextMenu.style.left = windowWidth - menuWidth + "px";
        }

        if ((windowHeight - posy) < menuHeight) {
            contextMenu.style.top = windowHeight - menuHeight + "px";
        }
    }

    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {
        let currentItems = menuSpecs.get(targetElementSelector);
        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        let newItem = {
            id: targetElementSelector + '_' + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true
        };
        currentItems.push(newItem);
        return newItem['id'];
    }

    function removeContextMenuOption(uid) {
        menuSpecs.forEach(function(v) {
            let index = -1;
            v.forEach(function(e, ei) {
                if (e['id'] == uid) {
                    index = ei;
                }
            });
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }
        gradioApp().addEventListener("click", function(e) {
            if (!e.isTrusted) return;
            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        gradioApp().addEventListener("contextmenu", function(e) {
            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) oldMenu.remove();
            menuSpecs.forEach(function(v, k) {
                if (e.composedPath()[0].matches(k)) {
                    showContextMenu(e, e.composedPath()[0], v);
                    e.preventDefault();
                }
            });
        });
        eventListenerApplied = true;
    }

    // 全局函数（供其他脚本调用）
    window.appendContextMenuOption = appendContextMenuOption;
    window.removeContextMenuOption = removeContextMenuOption;
    window.addContextMenuEventListener = addContextMenuEventListener;

    // 原有的 cancelGenerateForever（用于停止无限生成）
    window.cancelGenerateForever = function() {
        if (window.generateOnRepeatInterval) {
            clearInterval(window.generateOnRepeatInterval);
            window.generateOnRepeatInterval = null;
        }
    };

    // 添加自定义菜单项（Generate forever）
    function generateOnRepeat(genbuttonid, interruptbuttonid) {
        let genbutton = gradioApp().querySelector(genbuttonid);
        let interruptbutton = gradioApp().querySelector(interruptbuttonid);
        if (!interruptbutton || !interruptbutton.offsetParent) {
            if (genbutton) genbutton.click();
        }
        if (window.generateOnRepeatInterval) clearInterval(window.generateOnRepeatInterval);
        window.generateOnRepeatInterval = setInterval(function() {
            let intBtn = gradioApp().querySelector(interruptbuttonid);
            if (!intBtn || !intBtn.offsetParent) {
                let genBtn = gradioApp().querySelector(genbuttonid);
                if (genBtn) genBtn.click();
            }
        }, 500);
    }

    function generateOnRepeatForButtons() {
        generateOnRepeat('#generate_button', '#stop_button');
    }

    // 等待目标按钮出现后再绑定菜单选项和事件监听
    async function initContextMenu() {
        // 等待至少一个目标元素出现（generate_button 或 stop_button）
        await Promise.race([
            waitForElm('#generate_button'),
            waitForElm('#stop_button')
        ]);
        // 添加右键菜单选项
        appendContextMenuOption('#generate_button', 'Generate forever', generateOnRepeatForButtons);
        appendContextMenuOption('#stop_button', 'Generate forever', generateOnRepeatForButtons);
        // 添加全局右键事件监听
        addContextMenuEventListener();
        console.log("Context menu initialized (Gradio 6 compatible)");
    }

    // 开始初始化
    initContextMenu().catch(err => console.warn("Context menu init failed:", err));
})();