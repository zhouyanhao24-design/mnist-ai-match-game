from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pygame
import torch
from PIL import Image
from torchvision import datasets, transforms

from game_logic import MatchGameLogic, Tile
from models import build_model
from utils import get_device, load_checkpoint, set_seed


CELL_SIZE = 76
IMAGE_SIZE = 52
MARGIN = 16
TOP_OFFSET = 70
TRAY_TOP = 610
BACKGROUND = (245, 247, 250)
CARD = (255, 255, 255)
CARD_BORDER = (70, 90, 120)
TRAY_BG = (225, 232, 242)
TEXT = (30, 40, 55)


class DigitPredictor:
    def __init__(self, model_name: str, checkpoint: str | None, error_rate: float, device: torch.device) -> None:
        self.error_rate = error_rate
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.model = build_model(model_name).to(device)
        self.ready = False
        if checkpoint and Path(checkpoint).exists():
            load_checkpoint(self.model, checkpoint, device)
            self.model.eval()
            self.ready = True
        else:
            print("Warning: checkpoint not found. The game will use true labels for demonstration.")

    @torch.no_grad()
    def predict(self, image_array: np.ndarray, true_label: int) -> int:
        if self.ready:
            image = Image.fromarray(image_array.astype(np.uint8), mode="L")
            x = self.transform(image).unsqueeze(0).to(self.device)
            pred = int(self.model(x).argmax(dim=1).item())
        else:
            pred = int(true_label)

        if random.random() < self.error_rate:
            choices = [i for i in range(10) if i != pred]
            pred = random.choice(choices)
        return pred


def load_raw_mnist(data_dir: str):
    return datasets.MNIST(root=data_dir, train=False, download=True, transform=None)


def build_tiles(dataset, num_triples: int, seed: int) -> list[Tile]:
    random.seed(seed)
    label_to_indices: dict[int, list[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        if len(label_to_indices[int(label)]) < 100:
            label_to_indices[int(label)].append(idx)
        if all(len(v) >= 100 for v in label_to_indices.values()):
            break

    labels = [i % 10 for i in range(num_triples)]
    tiles: list[Tile] = []
    tile_id = 0
    for label in labels:
        chosen = random.sample(label_to_indices[label], 3)
        for idx in chosen:
            image, true_label = dataset[idx]
            tiles.append(
                Tile(
                    tile_id=tile_id,
                    true_label=int(true_label),
                    image=np.array(image),
                    row=0,
                    col=0,
                )
            )
            tile_id += 1

    random.shuffle(tiles)
    cols = 10
    for i, tile in enumerate(tiles):
        tile.row = i // cols
        tile.col = i % cols
    return tiles


def image_to_surface(image_array: np.ndarray) -> pygame.Surface:
    image = Image.fromarray(image_array.astype(np.uint8), mode="L").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = Image.merge("RGB", (image, image, image))
    return pygame.image.fromstring(rgb.tobytes(), rgb.size, rgb.mode)


def draw_text(screen, font, text: str, x: int, y: int) -> None:
    surface = font.render(text, True, TEXT)
    screen.blit(surface, (x, y))


def tile_rect(tile: Tile) -> pygame.Rect:
    x = MARGIN + tile.col * CELL_SIZE
    y = TOP_OFFSET + tile.row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE - 8, CELL_SIZE - 8)


def draw_tile(screen, font, tile: Tile, hidden: bool) -> None:
    rect = tile_rect(tile)
    pygame.draw.rect(screen, CARD, rect, border_radius=10)
    pygame.draw.rect(screen, CARD_BORDER, rect, 2, border_radius=10)
    if hidden and tile.predicted_label is None:
        pygame.draw.rect(screen, (20, 24, 30), rect.inflate(-12, -12), border_radius=8)
    else:
        surface = image_to_surface(tile.image)
        screen.blit(surface, (rect.x + 8, rect.y + 8))
    if tile.predicted_label is not None:
        draw_text(screen, font, str(tile.predicted_label), rect.x + 6, rect.y + 4)


def draw_tray(screen, font, logic: MatchGameLogic) -> None:
    tray_rect = pygame.Rect(MARGIN, TRAY_TOP, 950, 92)
    pygame.draw.rect(screen, TRAY_BG, tray_rect, border_radius=12)
    pygame.draw.rect(screen, CARD_BORDER, tray_rect, 2, border_radius=12)
    draw_text(screen, font, "Tray", MARGIN + 10, TRAY_TOP + 8)

    for i, tile in enumerate(logic.tray):
        x = MARGIN + 80 + i * 72
        y = TRAY_TOP + 25
        rect = pygame.Rect(x, y, 60, 60)
        pygame.draw.rect(screen, CARD, rect, border_radius=8)
        pygame.draw.rect(screen, CARD_BORDER, rect, 1, border_radius=8)
        if tile.predicted_label is not None:
            draw_text(screen, font, str(tile.predicted_label), x + 22, y + 18)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MNIST AI match game.")
    parser.add_argument("--model", choices=["resnet", "vgg"], default="resnet")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--num-triples", type=int, default=10)
    parser.add_argument("--tray-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    pygame.init()
    screen = pygame.display.set_mode((1000, 760))
    pygame.display.set_caption("MNIST AI Match Game")
    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 18)
    clock = pygame.time.Clock()

    dataset = load_raw_mnist(args.data_dir)
    tiles = build_tiles(dataset, args.num_triples, args.seed)
    logic = MatchGameLogic(tray_size=args.tray_size)
    checkpoint = args.checkpoint or f"checkpoints/{args.model}_mnist.pth"
    predictor = DigitPredictor(args.model, checkpoint, args.error_rate, device)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                tiles = build_tiles(dataset, args.num_triples, random.randint(0, 999999))
                logic.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                for tile in tiles:
                    if logic.can_select(tile) and tile_rect(tile).collidepoint(mx, my):
                        pred = predictor.predict(tile.image, tile.true_label)
                        remaining = sum(1 for t in tiles if not t.removed and not t.in_tray) - 1
                        logic.select_tile(tile, pred, remaining)
                        break

        screen.fill(BACKGROUND)
        draw_text(screen, font, "MNIST AI Match Game", MARGIN, 18)
        status = f"Model: {args.model} | Error rate: {args.error_rate:.0%} | Moves: {logic.moves} | Matches: {logic.matches} | R: restart"
        draw_text(screen, small_font, status, MARGIN, 46)

        for tile in tiles:
            if not tile.removed and not tile.in_tray:
                draw_tile(screen, small_font, tile, args.hidden)

        draw_tray(screen, font, logic)
        draw_text(screen, font, logic.message, MARGIN, 715)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
