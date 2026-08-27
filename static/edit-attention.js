// attention_edit.js - Gradio 6 兼容版（等待元素就绪后启用快捷键）
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

    function updateInput(target) {
        let e = new Event("input", { bubbles: true });
        Object.defineProperty(e, "target", { value: target });
        target.dispatchEvent(e);
    }

    function keyupEditAttention(event) {
        let target = event.originalTarget || event.composedPath()[0];
        // 匹配 prompt 相关的 textarea（兼容 Gradio 6 的类名和 ID）
        if (!target.matches("*:is([id*='_prompt'], .prompt) textarea")) return;
        if (!(event.metaKey || event.ctrlKey)) return;

        let isPlus = event.key == "ArrowUp";
        let isMinus = event.key == "ArrowDown";
        if (!isPlus && !isMinus) return;

        let selectionStart = target.selectionStart;
        let selectionEnd = target.selectionEnd;
        let text = target.value;

        function selectCurrentParenthesisBlock(OPEN, CLOSE) {
            if (selectionStart !== selectionEnd) return false;
            const before = text.substring(0, selectionStart);
            let beforeParen = before.lastIndexOf(OPEN);
            if (beforeParen == -1) return false;
            let beforeParenClose = before.lastIndexOf(CLOSE);
            while (beforeParenClose !== -1 && beforeParenClose > beforeParen) {
                beforeParen = before.lastIndexOf(OPEN, beforeParen - 1);
                beforeParenClose = before.lastIndexOf(CLOSE, beforeParenClose - 1);
            }
            const after = text.substring(selectionStart);
            let afterParen = after.indexOf(CLOSE);
            if (afterParen == -1) return false;
            let afterParenOpen = after.indexOf(OPEN);
            while (afterParenOpen !== -1 && afterParen > afterParenOpen) {
                afterParen = after.indexOf(CLOSE, afterParen + 1);
                afterParenOpen = after.indexOf(OPEN, afterParenOpen + 1);
            }
            if (beforeParen === -1 || afterParen === -1) return false;
            const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
            const lastColon = parenContent.lastIndexOf(":");
            selectionStart = beforeParen + 1;
            selectionEnd = selectionStart + lastColon;
            target.setSelectionRange(selectionStart, selectionEnd);
            return true;
        }

        function selectCurrentWord() {
            if (selectionStart !== selectionEnd) return false;
            const delimiters = ".,\\/!?%^*;:{}=`~() \r\n\t";
            while (!delimiters.includes(text[selectionStart - 1]) && selectionStart > 0) {
                selectionStart--;
            }
            while (!delimiters.includes(text[selectionEnd]) && selectionEnd < text.length) {
                selectionEnd++;
            }
            target.setSelectionRange(selectionStart, selectionEnd);
            return true;
        }

        if (!selectCurrentParenthesisBlock('<', '>') && !selectCurrentParenthesisBlock('(', ')')) {
            selectCurrentWord();
        }

        event.preventDefault();

        var closeCharacter = ')';
        var delta = 0.1;

        if (selectionStart > 0 && text[selectionStart - 1] == '<') {
            closeCharacter = '>';
            delta = 0.05;
        } else if (selectionStart == 0 || text[selectionStart - 1] != "(") {
            while (selectionEnd > selectionStart && text[selectionEnd - 1] == ' ') {
                selectionEnd -= 1;
            }
            if (selectionStart == selectionEnd) {
                return;
            }
            text = text.slice(0, selectionStart) + "(" + text.slice(selectionStart, selectionEnd) + ":1.0)" + text.slice(selectionEnd);
            selectionStart += 1;
            selectionEnd += 1;
        }

        var end = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
        var weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + 1 + end));
        if (isNaN(weight)) return;

        weight += isPlus ? delta : -delta;
        weight = parseFloat(weight.toPrecision(12));
        if (String(weight).length == 1) weight += ".0";

        if (closeCharacter == ')' && weight == 1) {
            var endParenPos = text.substring(selectionEnd).indexOf(')');
            text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + endParenPos + 1);
            selectionStart--;
            selectionEnd--;
        } else {
            text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + end);
        }

        target.focus();
        target.value = text;
        target.selectionStart = selectionStart;
        target.selectionEnd = selectionEnd;

        updateInput(target);
    }

    // 等待至少一个匹配的 textarea 出现后再绑定全局键盘事件
    waitForElm("*:is([id*='_prompt'], .prompt) textarea").then(() => {
        window.addEventListener('keydown', (event) => {
            keyupEditAttention(event);
        });
        console.log("Attention edit hotkeys enabled (Ctrl+↑/↓)");
    }).catch(err => console.warn("Attention edit init failed:", err));
})();