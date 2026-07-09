"""
ANGRY CALCULUS
==============
Un "Angry Birds" donde la trayectoria del pájaro NO se controla con el mouse,
sino que TÚ escribes directamente la función f(x) que define el vuelo.

Ejemplos válidos que puedes escribir:
    -0.15*x**2 + 3*x
    -0.2x^2 + 4x            (multiplicación implícita y ^ también funcionan)
    2*sin(x) + x
    -0.05*x**2 + 2*x + 1
    5*exp(-0.2*x)*sin(x)

Conceptos de cálculo diferencial usados de verdad (no de adorno):

  - La curva que ves ES f(x): el pájaro literalmente vuela sobre tu función.
  - f'(x) se calcula simbólicamente con sympy en cuanto escribes la función.
  - f'(0) es la pendiente inicial: la dirección con la que "sale disparado"
    el pájaro (se dibuja como la recta tangente amarilla en x=0).
  - Con la tecla TAB se abre una tabla que muestra cómo el cociente
    incremental (f(0+h) - f(0)) / h converge a f'(0) cuando h -> 0,
    es decir, la definición formal de derivada como límite.

Controles:
  (escribe)  -> vas armando el texto de tu función f(x)
  BACKSPACE  -> borra el último carácter
  ENTER      -> intenta lanzar con la función actual (si es válida)
  TAB        -> muestra / oculta la tabla de límites
  R          -> continuar tras un intento / reiniciar
  ESC        -> salir (y exporta los CSV)

Requisitos: pygame, sympy, pandas, numpy
    pip install pygame sympy pandas numpy
"""

import math
import pygame
import pymunk
import sympy as sp
import pandas as pd
import numpy as np
from mpmath.libmp import bin_to_radix
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from sympy.stats import density


WIDTH, HEIGHT = 960, 640
GROUND_Y = HEIGHT - 70
ORIGIN_X = 90
SCALE_X = 45
SCALE_Y = 45

FPS = 60
BIRD_RADIUS = 12
PIG_RADIUS = 20
X_MAX_FIELD = 19.0

COLOR_BG_TOP = (135, 206, 235)
COLOR_BG_BOTTOM = (200, 235, 245)
COLOR_GROUND = (86, 156, 60)
COLOR_CURVE = (255, 255, 255)
COLOR_TANGENT = (255, 210, 0)
COLOR_BIRD = (200, 30, 30)
COLOR_PIG = (60, 170, 60)
COLOR_TEXT = (20, 20, 20)
COLOR_PANEL = (255, 255, 255)
COLOR_PANEL_BORDER = (40, 40, 40)
COLOR_ERROR = (170, 20, 20)
COLOR_INPUT_BG = (255, 255, 220)

x_sym = sp.symbols("x")
FUNCION_INICIAL = "0"

TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

LEVELS = pd.DataFrame([
    {"nivel": 1, "pig_x": 6.0,  "pig_y": 1.0, "intentos": 5},
    {"nivel": 2, "pig_x": 9.0,  "pig_y": 2.2, "intentos": 5},
    {"nivel": 3, "pig_x": 12.0, "pig_y": 3.0, "intentos": 4},
    {"nivel": 4, "pig_x": 8.0,  "pig_y": 5.0, "intentos": 4},
    {"nivel": 5, "pig_x": 15.5, "pig_y": 1.5, "intentos": 3},
    {"nivel": 6, "pig_x": 23.7, "pig_y": 4.6, "intentos": 4}
])


def parse_funcion(texto):

    texto = texto.strip()
    if not texto:
        return None
    try:
        expr = parse_expr(texto, local_dict={"x": x_sym}, transformations=TRANSFORMACIONES)
    except Exception:
        return None

    libres = expr.free_symbols
    if libres - {x_sym}:
        return None

    try:
        deriv = sp.diff(expr, x_sym)
        f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
        fprime_num = sp.lambdify(x_sym, deriv, modules=["numpy"])
        _ = float(f_num(1.23))
        _ = float(fprime_num(1.23))
    except Exception:
        return None

    try:
        offset = float(f_num(0.0))
        if math.isnan(offset) or math.isinf(offset):
            offset = 0.0
    except Exception:
        offset = 0.0

    def g_num(xv, _f=f_num, _off=offset):
        return _f(xv) - _off

    return {
        "expr": expr,
        "expr_str": str(expr),
        "deriv": deriv,
        "deriv_str": str(deriv),
        "f": f_num,
        "fprime": fprime_num,
        "g": g_num,
        "offset": offset,
    }


def limite_derivada(funcion, x0, hs=(1.0, 0.1, 0.01, 0.001, 0.0001)):

    filas = []
    for h in hs:
        try:
            cociente = (funcion["f"](x0 + h) - funcion["f"](x0)) / h
        except Exception:
            cociente = float("nan")
        filas.append({"h": h, "cociente_incremental": cociente})
    df = pd.DataFrame(filas)
    try:
        df["f'(x0)_real"] = float(funcion["fprime"](x0))
    except Exception:
        df["f'(x0)_real"] = float("nan")
    return df


def to_screen(x, y):
    sx = ORIGIN_X + x * SCALE_X
    sy = GROUND_Y - y * SCALE_Y
    return sx, sy


class TrajectoryLogger:

    def __init__(self):
        self.rows = []

    def log_point(self, nivel, intento, funcion_texto, t, x, y, pendiente):
        self.rows.append({
            "nivel": nivel, "intento": intento, "funcion": funcion_texto,
            "t": t, "x": x, "y": y, "pendiente": pendiente,
        })

    def to_dataframe(self):
        return pd.DataFrame(self.rows)

    def export_csv(self, path="trayectorias.csv"):
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(path, index=False)
        return df


class ScoreBoard:
    def __init__(self):
        self.rows = []
        self.puntaje_total = 0

    def registrar(self, nivel, funcion_texto, intentos_usados, intentos_totales, exito):
        puntos = 0
        if exito:
            puntos = 100 + (intentos_totales - intentos_usados) * 20
        self.puntaje_total += puntos
        self.rows.append({
            "nivel": nivel, "funcion": funcion_texto, "exito": exito,
            "intentos_usados": intentos_usados, "puntos": puntos,
            "puntaje_acumulado": self.puntaje_total,
        })

    def to_dataframe(self):
        return pd.DataFrame(self.rows)

    def export_csv(self, path="puntajes.csv"):
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(path, index=False)
        return df


AJUSTANDO, LANZANDO, RESULTADO, FIN_JUEGO = range(4)


class Game:
    def __init__(self):
        pygame.init()
        pygame.key.start_text_input()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Angry Calculus - escribe tu propia f(x)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_big = pygame.font.SysFont("consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_input = pygame.font.SysFont("consolas", 22, bold=True)

        self.nivel_idx = 0
        self.intentos_restantes = int(LEVELS.iloc[0]["intentos"])
        self.estado = AJUSTANDO
        self.mostrar_limites = False
        self.mensaje_resultado = ""
        self.intento_num = 0

        self.buffer = FUNCION_INICIAL
        self.funcion_actual = parse_funcion(self.buffer)
        self.error_msg = ""

        self.logger = TrajectoryLogger()
        self.scoreboard = ScoreBoard()

        self.t = 0.0
        self.v = 6.0
        self.bird_x = 0.0
        self.bird_y = 0.0

    @property
    def nivel_actual(self):
        return LEVELS.iloc[self.nivel_idx]

    def reset_intentos_nivel(self):
        self.intentos_restantes = int(self.nivel_actual["intentos"])

    def siguiente_nivel(self):
        self.nivel_idx += 1
        if self.nivel_idx >= len(LEVELS):
            self.estado = FIN_JUEGO
        else:
            self.reset_intentos_nivel()
            self.estado = AJUSTANDO

    def actualizar_buffer(self, nuevo_texto):
        self.buffer = nuevo_texto
        resultado = parse_funcion(self.buffer)
        if resultado is not None:
            self.funcion_actual = resultado
            self.error_msg = ""
        else:
            self.error_msg = "Función inválida (usando la última válida)"

    def intentar_lanzar(self):
        resultado = parse_funcion(self.buffer)
        if resultado is None:
            self.error_msg = "No se puede lanzar: función inválida"
            return
        self.funcion_actual = resultado
        self.error_msg = ""
        self.lanzar()

    def lanzar(self):
        self.estado = LANZANDO
        self.t = 0.0
        self.bird_x = 0.0
        self.bird_y = 0.0
        self.intento_num += 1
        self.intentos_restantes -= 1

    def actualizar_lanzamiento(self, dt):
        self.t += dt
        self.bird_x = self.v * self.t
        try:
            self.bird_y = float(self.funcion_actual["g"](self.bird_x))
            pendiente = float(self.funcion_actual["fprime"](self.bird_x))
        except Exception:
            self.terminar_lanzamiento(exito=False)
            return

        if math.isnan(self.bird_y) or math.isinf(self.bird_y):
            self.terminar_lanzamiento(exito=False)
            return

        self.logger.log_point(
            int(self.nivel_actual["nivel"]), self.intento_num, self.buffer,
            round(self.t, 3), round(self.bird_x, 3),
            round(self.bird_y, 3), round(pendiente, 3),
        )

        pig_x = self.nivel_actual["pig_x"]
        pig_y = self.nivel_actual["pig_y"]
        dist = math.hypot(self.bird_x - pig_x, self.bird_y - pig_y)

        if dist < (PIG_RADIUS + BIRD_RADIUS) / SCALE_X * 1.3:
            self.terminar_lanzamiento(exito=True)
            return

        if self.bird_y < -0.3 or self.bird_x > X_MAX_FIELD:
            self.terminar_lanzamiento(exito=False)
            return

    def terminar_lanzamiento(self, exito):
        nivel_num = int(self.nivel_actual["nivel"])
        intentos_totales = int(self.nivel_actual["intentos"])
        usados = intentos_totales - self.intentos_restantes

        if exito:
            self.mensaje_resultado = "¡Le diste al cerdo! Nivel superado."
            self.scoreboard.registrar(nivel_num, self.buffer, usados, intentos_totales, True)
            self.estado = RESULTADO
        else:
            if self.intentos_restantes <= 0:
                self.mensaje_resultado = "Sin intentos. Reiniciando nivel."
                self.scoreboard.registrar(nivel_num, self.buffer, usados, intentos_totales, False)
                self.reset_intentos_nivel()
            else:
                self.mensaje_resultado = f"Fallaste. Intentos restantes: {self.intentos_restantes}"
            self.estado = RESULTADO

    def continuar_desde_resultado(self):
        if "superado" in self.mensaje_resultado:
            self.siguiente_nivel()
        else:
            self.estado = AJUSTANDO

    def dibujar_fondo(self):
        for i in range(HEIGHT):
            r = COLOR_BG_TOP[0] + (COLOR_BG_BOTTOM[0] - COLOR_BG_TOP[0]) * i // HEIGHT
            g = COLOR_BG_TOP[1] + (COLOR_BG_BOTTOM[1] - COLOR_BG_TOP[1]) * i // HEIGHT
            b_ = COLOR_BG_TOP[2] + (COLOR_BG_BOTTOM[2] - COLOR_BG_TOP[2]) * i // HEIGHT
            pygame.draw.line(self.screen, (r, g, b_), (0, i), (WIDTH, i))
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))

    def dibujar_curva_preview(self):
        funcion = self.funcion_actual
        puntos = []
        for xv in np.linspace(0, X_MAX_FIELD, 250):
            try:
                yv = float(funcion["g"](xv))
            except Exception:
                break
            if math.isnan(yv) or math.isinf(yv):
                break
            puntos.append(to_screen(xv, yv))
            if yv < -1:
                break
        if len(puntos) > 1:
            pygame.draw.lines(self.screen, COLOR_CURVE, False, puntos, 3)


        try:
            pendiente0 = float(funcion["fprime"](0.0))
            p1 = to_screen(0, 0)
            p2 = to_screen(1.3, pendiente0 * 1.3)
            pygame.draw.line(self.screen, COLOR_TANGENT, p1, p2, 3)
        except Exception:
            pass

    def dibujar_pig(self):
        px, py = to_screen(self.nivel_actual["pig_x"], self.nivel_actual["pig_y"])
        pygame.draw.circle(self.screen, COLOR_PIG, (int(px), int(py)), PIG_RADIUS)
        pygame.draw.circle(self.screen, (20, 60, 20), (int(px), int(py)), PIG_RADIUS, 2)

    def dibujar_bird(self):
        bx, by = to_screen(self.bird_x, self.bird_y)
        pygame.draw.circle(self.screen, COLOR_BIRD, (int(bx), int(by)), BIRD_RADIUS)

    def dibujar_caja_texto(self):
        """Caja donde el jugador escribe f(x)."""
        box_rect = pygame.Rect(10, HEIGHT - 40, WIDTH - 20, 32)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, box_rect)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, box_rect, 2)
        texto = f"f(x) = {self.buffer}"
        surf = self.font_input.render(texto, True, (20, 20, 20))
        self.screen.blit(surf, (box_rect.x + 8, box_rect.y + 4))
        if (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = box_rect.x + 8 + surf.get_width() + 2
            pygame.draw.line(
                self.screen, (20, 20, 20),
                (cursor_x, box_rect.y + 4), (cursor_x, box_rect.y + 26), 2
            )

    def dibujar_hud(self):
        nivel = self.nivel_actual
        deriv_txt = self.funcion_actual["deriv_str"] if self.funcion_actual else "?"
        expr_txt = self.funcion_actual["expr_str"] if self.funcion_actual else "?"
        textos = [
            f"Nivel {int(nivel['nivel'])}/{len(LEVELS)}   Intentos: {self.intentos_restantes}   Puntaje: {self.scoreboard.puntaje_total}",
            f"f(x) = {expr_txt}",
            f"f'(x) = {deriv_txt}",
            "Escribe tu funcion abajo. ENTER: lanzar   TAB: tabla de limites",
        ]
        y = 8
        for txt in textos:
            surf = self.font.render(txt, True, COLOR_TEXT)
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4))
            bg.fill(COLOR_PANEL)
            bg.set_alpha(210)
            self.screen.blit(bg, (5, y - 2))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 4

        if self.funcion_actual and abs(self.funcion_actual["offset"]) > 1e-9:
            aviso = f"(el vuelo siempre sale de (0,0); tu f(x) se desplaza {-self.funcion_actual['offset']:+.2f} para lograrlo)"
            surf = self.font_small.render(aviso, True, (90, 90, 90))
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4))
            bg.fill(COLOR_PANEL)
            bg.set_alpha(200)
            self.screen.blit(bg, (5, y - 2))
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 4

        if self.error_msg:
            surf = self.font.render(self.error_msg, True, COLOR_ERROR)
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4))
            bg.fill(COLOR_PANEL)
            bg.set_alpha(220)
            self.screen.blit(bg, (5, y - 2))
            self.screen.blit(surf, (10, y))

    def dibujar_tabla_limites(self):
        if not self.mostrar_limites or self.funcion_actual is None:
            return
        df = limite_derivada(self.funcion_actual, 0.0)
        x0_panel, y0_panel = WIDTH - 320, 10
        titulo = self.font.render("lim h->0 (f(0+h)-f(0))/h", True, COLOR_TEXT)
        panel_h = 26 * (len(df) + 2)
        bg = pygame.Surface((310, panel_h))
        bg.fill(COLOR_PANEL)
        bg.set_alpha(230)
        self.screen.blit(bg, (x0_panel, y0_panel))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (x0_panel, y0_panel, 310, panel_h), 1)
        self.screen.blit(titulo, (x0_panel + 8, y0_panel + 4))
        row_y = y0_panel + 30
        for _, row in df.iterrows():
            linea = f"h={row['h']:<8} -> {row['cociente_incremental']:.4f}"
            surf = self.font_small.render(linea, True, COLOR_TEXT)
            self.screen.blit(surf, (x0_panel + 8, row_y))
            row_y += 22
        final = self.font_small.render(
            f"f'(0) real = {df['f\'(x0)_real'].iloc[0]:.4f}", True, (150, 0, 0)
        )
        self.screen.blit(final, (x0_panel + 8, row_y + 4))

    def dibujar_mensaje_centrado(self, texto, sub=""):
        surf = self.font_big.render(texto, True, (10, 10, 10))
        rect = surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
        bg = pygame.Surface((rect.width + 40, rect.height + 70))
        bg.fill(COLOR_PANEL)
        bg.set_alpha(235)
        self.screen.blit(bg, (rect.x - 20, rect.y - 20))
        self.screen.blit(surf, rect)
        if sub:
            surf2 = self.font.render(sub, True, (60, 60, 60))
            rect2 = surf2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
            self.screen.blit(surf2, rect2)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.TEXTINPUT and self.estado == AJUSTANDO:
                    self.actualizar_buffer(self.buffer + event.text)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        self.mostrar_limites = not self.mostrar_limites
                    elif event.key == pygame.K_BACKSPACE and self.estado == AJUSTANDO:
                        self.actualizar_buffer(self.buffer[:-1])
                    elif event.key == pygame.K_RETURN and self.estado == AJUSTANDO:
                        self.intentar_lanzar()
                    elif event.key == pygame.K_r and self.estado == RESULTADO:
                        self.continuar_desde_resultado()
                    elif event.key == pygame.K_r and self.estado == FIN_JUEGO:
                        self.__init__()

            if self.estado == LANZANDO:
                self.actualizar_lanzamiento(dt)

            self.dibujar_fondo()
            self.dibujar_pig()

            if self.estado == AJUSTANDO:
                self.dibujar_curva_preview()
                self.dibujar_caja_texto()
            if self.estado == LANZANDO:
                self.dibujar_bird()

            self.dibujar_hud()
            self.dibujar_tabla_limites()

            if self.estado == RESULTADO:
                self.dibujar_mensaje_centrado(
                    self.mensaje_resultado, "Presiona R para continuar"
                )
            if self.estado == FIN_JUEGO:
                self.dibujar_mensaje_centrado(
                    f"¡Juego completado! Puntaje final: {self.scoreboard.puntaje_total}",
                    "Presiona R para reiniciar o ESC para salir",
                )

            pygame.display.flip()

        df_traj = self.logger.export_csv("trayectorias.csv")
        df_scores = self.scoreboard.export_csv("puntajes.csv")
        print("\n--- Exportado con pandas ---")
        print(f"trayectorias.csv -> {len(df_traj)} puntos registrados")
        print(f"puntajes.csv -> {len(df_scores)} intentos registrados")
        pygame.quit()


if __name__ == "__main__":
    Game().run()