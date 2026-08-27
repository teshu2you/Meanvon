// lightbox.js - Gradio 6 兼容版（等待元素就绪后初始化灯箱）
(function() {
    // 确保 waitForElm 存在
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

    // Gradio 6 中 gradioApp() 直接返回 document（或可以返回 document.body，但保持与原有代码兼容）
    function gradioApp() {
        return document;
    }

    // ----- 原有函数（大部分保持不变，只修改了部分选择器或兼容性） -----
    function closeModal() {
        const modal = gradioApp().getElementById("lightboxModal");
        if (modal) modal.style.display = "none";
    }

    function showModal(event) {
        const source = event.target || event.srcElement;
        const modalImage = gradioApp().getElementById("modalImage");
        const lb = gradioApp().getElementById("lightboxModal");
        if (!modalImage || !lb) return;
        modalImage.src = source.src;
        if (modalImage.style.display === 'none') {
            lb.style.setProperty('background-image', 'url(' + source.src + ')');
        }
        lb.style.display = "flex";
        lb.focus();
        event.stopPropagation();
    }

    function negmod(n, m) {
        return ((n % m) + m) % m;
    }

    function updateOnBackgroundChange() {
        const modalImage = gradioApp().getElementById("modalImage");
        if (modalImage && modalImage.offsetParent) {
            let currentButton = selected_gallery_button();
            if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
                modalImage.src = currentButton.children[0].src;
                if (modalImage.style.display === 'none') {
                    const modal = gradioApp().getElementById("lightboxModal");
                    if (modal) modal.style.setProperty('background-image', `url(${modalImage.src})`);
                }
            }
        }
    }

    function all_gallery_buttons() {
        // 选择器：匹配 Gallery 组件中的缩略图项（保留原始类名，Gradio 6 中可能仍存在）
        var allGalleryButtons = gradioApp().querySelectorAll('.image_gallery .thumbnails > .thumbnail-item.thumbnail-small');
        var visibleGalleryButtons = [];
        allGalleryButtons.forEach(function(elem) {
            if (elem.parentElement && elem.parentElement.offsetParent) {
                visibleGalleryButtons.push(elem);
            }
        });
        return visibleGalleryButtons;
    }

    function selected_gallery_button() {
        return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
    }

    function selected_gallery_index() {
        return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
    }

    function modalImageSwitch(offset) {
        var galleryButtons = all_gallery_buttons();
        if (galleryButtons.length > 1) {
            var currentButton = selected_gallery_button();
            var result = -1;
            galleryButtons.forEach(function(v, i) {
                if (v == currentButton) result = i;
            });
            if (result != -1) {
                var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
                nextButton.click();
                const modalImage = gradioApp().getElementById("modalImage");
                const modal = gradioApp().getElementById("lightboxModal");
                if (modalImage && modal) {
                    modalImage.src = nextButton.children[0].src;
                    if (modalImage.style.display === 'none') {
                        modal.style.setProperty('background-image', `url(${modalImage.src})`);
                    }
                }
                setTimeout(function() { if (modal) modal.focus(); }, 10);
            }
        }
    }

    function saveImage() {
        // 原有 saveImage 逻辑为空，可保留或扩展
    }

    function modalSaveImage(event) {
        event.stopPropagation();
    }

    function modalNextImage(event) {
        modalImageSwitch(1);
        event.stopPropagation();
    }

    function modalPrevImage(event) {
        modalImageSwitch(-1);
        event.stopPropagation();
    }

    function modalKeyHandler(event) {
        switch (event.key) {
            case "s":
                saveImage();
                break;
            case "ArrowLeft":
                modalPrevImage(event);
                break;
            case "ArrowRight":
                modalNextImage(event);
                break;
            case "Escape":
                closeModal();
                break;
        }
    }

    function setupImageForLightbox(e) {
        if (e.dataset.modded) return;
        e.dataset.modded = true;
        e.style.cursor = 'pointer';
        e.style.userSelect = 'none';

        var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
        var eventType = isFirefox ? 'mousedown' : 'click';

        e.addEventListener(eventType, function(evt) {
            if (evt.button == 1) {
                open(evt.target.src);
                evt.preventDefault();
                return;
            }
            if (evt.button != 0) return;
            modalZoomSet(gradioApp().getElementById('modalImage'), true);
            evt.preventDefault();
            showModal(evt);
        }, true);
    }

    function modalZoomSet(modalImage, enable) {
        if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
    }

    function modalZoomToggle(event) {
        var modalImage = gradioApp().getElementById("modalImage");
        modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
        event.stopPropagation();
    }

    function modalTileImageToggle(event) {
        const modalImage = gradioApp().getElementById("modalImage");
        const modal = gradioApp().getElementById("lightboxModal");
        if (!modalImage || !modal) return;
        const isTiling = modalImage.style.display === 'none';
        if (isTiling) {
            modalImage.style.display = 'block';
            modal.style.setProperty('background-image', 'none');
        } else {
            modalImage.style.display = 'none';
            modal.style.setProperty('background-image', `url(${modalImage.src})`);
        }
        event.stopPropagation();
    }

    // ----- 初始化灯箱 DOM（仅创建一次）-----
    let lightboxInitialized = false;

    function initLightboxDOM() {
        if (lightboxInitialized) return;
        lightboxInitialized = true;

        const modal = document.createElement('div');
        modal.onclick = closeModal;
        modal.id = "lightboxModal";
        modal.tabIndex = 0;
        modal.addEventListener('keydown', modalKeyHandler, true);

        const modalControls = document.createElement('div');
        modalControls.className = 'modalControls gradio-container';
        modal.append(modalControls);

        const modalZoom = document.createElement('span');
        modalZoom.className = 'modalZoom cursor';
        modalZoom.innerHTML = '&#10529;';
        modalZoom.addEventListener('click', modalZoomToggle, true);
        modalZoom.title = "Toggle zoomed view";
        modalControls.appendChild(modalZoom);

        // 可选：平铺和保存按钮（如需要可取消注释）
        // const modalTileImage = document.createElement('span');
        // modalTileImage.className = 'modalTileImage cursor';
        // modalTileImage.innerHTML = '&#8862;';
        // modalTileImage.addEventListener('click', modalTileImageToggle, true);
        // modalTileImage.title = "Preview tiling";
        // modalControls.appendChild(modalTileImage);
        //
        // const modalSave = document.createElement("span");
        // modalSave.className = "modalSave cursor";
        // modalSave.id = "modal_save";
        // modalSave.innerHTML = "&#x1F5AB;";
        // modalSave.addEventListener("click", modalSaveImage, true);
        // modalSave.title = "Save Image(s)";
        // modalControls.appendChild(modalSave);

        const modalClose = document.createElement('span');
        modalClose.className = 'modalClose cursor';
        modalClose.innerHTML = '&times;';
        modalClose.onclick = closeModal;
        modalClose.title = "Close image viewer";
        modalControls.appendChild(modalClose);

        const modalImage = document.createElement('img');
        modalImage.id = 'modalImage';
        modalImage.onclick = closeModal;
        modalImage.tabIndex = 0;
        modalImage.addEventListener('keydown', modalKeyHandler, true);
        modal.appendChild(modalImage);

        const modalPrev = document.createElement('a');
        modalPrev.className = 'modalPrev';
        modalPrev.innerHTML = '&#10094;';
        modalPrev.tabIndex = 0;
        modalPrev.addEventListener('click', modalPrevImage, true);
        modalPrev.addEventListener('keydown', modalKeyHandler, true);
        modal.appendChild(modalPrev);

        const modalNext = document.createElement('a');
        modalNext.className = 'modalNext';
        modalNext.innerHTML = '&#10095;';
        modalNext.tabIndex = 0;
        modalNext.addEventListener('click', modalNextImage, true);
        modalNext.addEventListener('keydown', modalKeyHandler, true);
        modal.appendChild(modalNext);

        // 将灯箱添加到 body（Gradio 6 中直接添加到 document.body）
        document.body.appendChild(modal);
    }

    // ----- 初始化灯箱功能：等待图库元素出现后绑定事件 -----
    async function initLightbox() {
        // 等待至少一个 .image_gallery 出现（表示图库已渲染）
        await waitForElm('.image_gallery');
        // 创建灯箱 DOM（如果尚未创建）
        initLightboxDOM();

        // 为当前所有图片和未来动态添加的图片设置点击事件
        function bindToCurrentImages() {
            const fullImgPreviews = gradioApp().querySelectorAll('.image_gallery > div > img');
            fullImgPreviews.forEach(setupImageForLightbox);
            updateOnBackgroundChange();
        }

        bindToCurrentImages();

        // 监听 DOM 变化，当新的图片被添加时自动绑定
        const observer = new MutationObserver(() => {
            bindToCurrentImages();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // 启动初始化
    initLightbox().catch(err => console.warn("Lightbox init failed:", err));
})();