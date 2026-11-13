# Test image sources

| File | Source |
|---|---|
| stop_sign.jpg, red_bus.jpg, cats_remotes.jpg, bedroom.jpg, bear.jpg, food_bowl.jpg, skier.jpg, teddy_bears.jpg, living_room.jpg, ribbon_cutting.jpg | COCO val2017 ids 724, 1584, 39769, 632, 285, 3501, 785, 776, 139, 5001 (Flickr, CC BY licences) |
| cat.png, landscape.png, person.png, astronaut.jpg | `huggingface/documentation-images` dataset, `diffusers/` folder |
| bus_injected.jpg | red_bus.jpg with an injected instruction drawn on it |
| inject_poster.png, inject_note.png | Made by us: text-only images with injected instructions |
| tiny_blurry.jpg | living_room.jpg shrunk to 24×18 at JPEG quality 10 |
| corrupted.jpg | First 2 KB of bear.jpg (truncated file) |
| empty.png | 0-byte file |
| not_an_image.png | A text file with a .png name |
