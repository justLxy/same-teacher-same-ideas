"""Redraw every paper figure from the committed result summaries: python3 -m figures.make_all"""
from figures import gen_fig_data_training, gen_fig_distances, gen_fig_positions, gen_fig_released

if __name__ == "__main__":
    for module in (gen_fig_distances, gen_fig_released, gen_fig_positions, gen_fig_data_training):
        print(module.main())
