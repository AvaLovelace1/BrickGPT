'use strict';

(function() {
    // Format video selector for compositional prompts.
    let captions = ["a DSLR photo of a squirrel  chopping vegetables",
"a DSLR photo of a squirrel  dancing",
"a DSLR photo of a squirrel  eating a hamburger",
"a DSLR photo of a squirrel  playing the saxophone",
"a DSLR photo of a squirrel  reading a book",
"a DSLR photo of a squirrel  ",
"a DSLR photo of a squirrel  riding a motorcycle",
"a DSLR photo of a squirrel  riding a skateboard",
"a DSLR photo of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a kimono chopping vegetables",
"a DSLR photo of a squirrel wearing a kimono dancing",
"a DSLR photo of a squirrel wearing a kimono eating a hamburger",
"a DSLR photo of a squirrel wearing a kimono playing the saxophone",
"a DSLR photo of a squirrel wearing a kimono reading a book",
"a DSLR photo of a squirrel wearing a kimono ",
"a DSLR photo of a squirrel wearing a kimono riding a motorcycle",
"a DSLR photo of a squirrel wearing a kimono riding a skateboard",
"a DSLR photo of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a kimono wielding a katana",
"a DSLR photo of a squirrel wearing a medieval suit of armor chopping vegetables",
"a DSLR photo of a squirrel wearing a medieval suit of armor dancing",
"a DSLR photo of a squirrel wearing a medieval suit of armor eating a hamburger",
"a DSLR photo of a squirrel wearing a medieval suit of armor playing the saxophone",
"a DSLR photo of a squirrel wearing a medieval suit of armor reading a book",
"a DSLR photo of a squirrel wearing a medieval suit of armor ",
"a DSLR photo of a squirrel wearing a medieval suit of armor riding a motorcycle",
"a DSLR photo of a squirrel wearing a medieval suit of armor riding a skateboard",
"a DSLR photo of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a medieval suit of armor wielding a katana",
"a DSLR photo of a squirrel wearing an elegant ballgown chopping vegetables",
"a DSLR photo of a squirrel wearing an elegant ballgown dancing",
"a DSLR photo of a squirrel wearing an elegant ballgown eating a hamburger",
"a DSLR photo of a squirrel wearing an elegant ballgown playing the saxophone",
"a DSLR photo of a squirrel wearing an elegant ballgown reading a book",
"a DSLR photo of a squirrel wearing an elegant ballgown ",
"a DSLR photo of a squirrel wearing an elegant ballgown riding a motorcycle",
"a DSLR photo of a squirrel wearing an elegant ballgown riding a skateboard",
"a DSLR photo of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing an elegant ballgown wielding a katana",
"a DSLR photo of a squirrel wearing a purple hoodie chopping vegetables",
"a DSLR photo of a squirrel wearing a purple hoodie dancing",
"a DSLR photo of a squirrel wearing a purple hoodie eating a hamburger",
"a DSLR photo of a squirrel wearing a purple hoodie playing the saxophone",
"a DSLR photo of a squirrel wearing a purple hoodie reading a book",
"a DSLR photo of a squirrel wearing a purple hoodie ",
"a DSLR photo of a squirrel wearing a purple hoodie riding a motorcycle",
"a DSLR photo of a squirrel wearing a purple hoodie riding a skateboard",
"a DSLR photo of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a purple hoodie wielding a katana",
"a DSLR photo of a squirrel  wielding a katana",
"a highly detailed metal sculpture of a squirrel  chopping vegetables",
"a highly detailed metal sculpture of a squirrel  dancing",
"a highly detailed metal sculpture of a squirrel  eating a hamburger",
"a highly detailed metal sculpture of a squirrel  playing the saxophone",
"a highly detailed metal sculpture of a squirrel  reading a book",
"a highly detailed metal sculpture of a squirrel  ",
"a highly detailed metal sculpture of a squirrel  riding a motorcycle",
"a highly detailed metal sculpture of a squirrel  riding a skateboard",
"a highly detailed metal sculpture of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a kimono chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a kimono dancing",
"a highly detailed metal sculpture of a squirrel wearing a kimono eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a kimono playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a kimono reading a book",
"a highly detailed metal sculpture of a squirrel wearing a kimono ",
"a highly detailed metal sculpture of a squirrel wearing a kimono riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a kimono riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a kimono wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor dancing",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor reading a book",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor ",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown dancing",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown reading a book",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown ",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie dancing",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie reading a book",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie ",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie wielding a katana",
"a highly detailed metal sculpture of a squirrel  wielding a katana",
"an intricate wooden carving of a squirrel  chopping vegetables",
"an intricate wooden carving of a squirrel  dancing",
"an intricate wooden carving of a squirrel  eating a hamburger",
"an intricate wooden carving of a squirrel  playing the saxophone",
"an intricate wooden carving of a squirrel  reading a book",
"an intricate wooden carving of a squirrel  ",
"an intricate wooden carving of a squirrel  riding a motorcycle",
"an intricate wooden carving of a squirrel  riding a skateboard",
"an intricate wooden carving of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a kimono chopping vegetables",
"an intricate wooden carving of a squirrel wearing a kimono dancing",
"an intricate wooden carving of a squirrel wearing a kimono eating a hamburger",
"an intricate wooden carving of a squirrel wearing a kimono playing the saxophone",
"an intricate wooden carving of a squirrel wearing a kimono reading a book",
"an intricate wooden carving of a squirrel wearing a kimono ",
"an intricate wooden carving of a squirrel wearing a kimono riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a kimono riding a skateboard",
"an intricate wooden carving of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a kimono wielding a katana",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor chopping vegetables",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor dancing",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor eating a hamburger",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor playing the saxophone",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor reading a book",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor ",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor riding a skateboard",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor wielding a katana",
"an intricate wooden carving of a squirrel wearing an elegant ballgown chopping vegetables",
"an intricate wooden carving of a squirrel wearing an elegant ballgown dancing",
"an intricate wooden carving of a squirrel wearing an elegant ballgown eating a hamburger",
"an intricate wooden carving of a squirrel wearing an elegant ballgown playing the saxophone",
"an intricate wooden carving of a squirrel wearing an elegant ballgown reading a book",
"an intricate wooden carving of a squirrel wearing an elegant ballgown ",
"an intricate wooden carving of a squirrel wearing an elegant ballgown riding a motorcycle",
"an intricate wooden carving of a squirrel wearing an elegant ballgown riding a skateboard",
"an intricate wooden carving of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing an elegant ballgown wielding a katana",
"an intricate wooden carving of a squirrel wearing a purple hoodie chopping vegetables",
"an intricate wooden carving of a squirrel wearing a purple hoodie dancing",
"an intricate wooden carving of a squirrel wearing a purple hoodie eating a hamburger",
"an intricate wooden carving of a squirrel wearing a purple hoodie playing the saxophone",
"an intricate wooden carving of a squirrel wearing a purple hoodie reading a book",
"an intricate wooden carving of a squirrel wearing a purple hoodie ",
"an intricate wooden carving of a squirrel wearing a purple hoodie riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a purple hoodie riding a skateboard",
"an intricate wooden carving of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a purple hoodie wielding a katana",
"an intricate wooden carving of a squirrel  wielding a katana"];

    let imagen_pieces = [
        ["a DSLR photo of a squirrel", "an intricate wooden carving of a squirrel", "a highly detailed metal sculpture of a squirrel"],
        ["", "wearing a kimono", "wearing a medieval suit of armor", "wearing a purple hoodie", "wearing an elegant ballgown"],
        ["", "reading a book", "riding a motorcycle", "playing the saxophone", "chopping vegetables", "sitting at a pottery wheel shaping a clay bowl",
        "riding a skateboard", "wielding a katana", "eating a hamburger", "dancing"],
    ];

    const updateCompositionVideo = () => {
        let phrase = "";
        for (let depth = 1; depth <= imagen_pieces.length; depth++) {
            let tagContainer = document.getElementById('compositional_tags_depth_' + depth);
            let selected = tagContainer.querySelectorAll('.selected');

            // Make sure at most one item is selected at this level.
            if (selected.length > 1) {
                // Too many tags selected at this level. Shouldn't have happened, but unselect them.
                selected.slice(1, selected.length).forEach((chunk) => {
                    chunk.classList = "";
                });
            }

            // Make sure at least one item is selected at this level.
            if (selected.length == 0) {
                tagContainer.querySelector('span').classList = "selected";
                updateCompositionVideo();
                return;
            }

            let segment = selected[0].getAttribute("data-segment");
            phrase = phrase + segment;
        }

        if (captions.includes(phrase)) {
            let compositionalVideo = document.getElementById('compositionalVideo');
            let container = compositionalVideo.parentNode;
            let videoName = phrase.replaceAll(' ', '_') + '_rgbdn_hq_15000.mp4';
            let sourceURL = "https://pub-b1f092b6867f4495b8f149d222a3bffe.r2.dev/journey_sept28/cropped/full_continuous/" + videoName;
            console.log('phrase found: ', phrase, sourceURL);

            let width = compositionalVideo.offsetWidth;
            let height = compositionalVideo.offsetHeight;

            // let oldSourceEl = compositionalVideo.querySelector('source');
            let oldSourceEl = document.getElementById('compositionalVideoSrc');
            oldSourceEl.src = sourceURL;

            container.style = "opacity: 0;";
            setTimeout(() => {
                container.style = "opacity: 1;";
                compositionalVideo.load();
            }, 750);

            let captionEl = document.getElementById('compositionalCaption');
            if (captionEl)
                captionEl.innerHTML = phrase;
        } else {
            console.log('phrase NOT found: ' + phrase);
        }
    };

    const deselect = (element) => {
        element.classList = ("" + element.classList).replace('selected', ' ');  // not clean, what about spaces?
    }

    const tagClicked = (event) => {
        event.target.parentNode.querySelectorAll('.selected').forEach(deselect);
        event.target.classList = "selected";
        updateCompositionVideo();
    };

    let phraseContainer = document.querySelector('.compositional .text');
    imagen_pieces.forEach((phrases, depth) => {
        depth = depth + 1;
        let tagContainer = document.createElement("P");
        tagContainer.classList = "selectable left";
        tagContainer.id = "compositional_tags_depth_" + depth;

        phrases.forEach((segment, i) => {
            if (depth > 1) {
                segment = " " + segment;
            }

            let tag = document.createElement("SPAN");
            let text = segment.trim();
            if (!text)
                text = '[...]';

            tag.appendChild(document.createTextNode(text));
            tag.setAttribute("data-segment", segment);
            tag.onclick = tagClicked;
            if (i == 0)
                tag.classList = "selected";

            tagContainer.appendChild(tag);
        });

        phraseContainer.appendChild(tagContainer);
    });

})();

(function() {
    // Click to load handlers for 3D meshes.
    document.querySelectorAll('button.loads-model').forEach((button) => {
        button.setAttribute('data-action', 'load');
        button.addEventListener('click', () => {
            let model = document.getElementById(button.getAttribute("data-controls"));

            if (button.getAttribute('data-action') == 'load') {
                model.dismissPoster();
                button.classList = "btn btn-disabled";
                button.innerHTML = "Hide 3D model";
                button.setAttribute('data-action', 'unload');
            } else {
                model.showPoster();
                button.classList = "btn btn-primary";
                button.innerHTML = "Load 3D model";
                button.setAttribute('data-action', 'load');
            }

        });
    });
})();