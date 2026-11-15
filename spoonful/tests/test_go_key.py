"""The GO button heard through Windows (io/go_key.py): one press = one GO, whatever has the keyboard."""
from spoonful.io.go_key import VK_NEXT, VK_PRIOR, GoKey


def test_one_press_is_one_go_and_bounce_is_ignored():
    keys, got = set(), []
    k = GoKey(lambda: got.append(1), state_fn=lambda vk: vk in keys)
    down = k.poll(False, 10.0)
    assert not down and not got
    keys.add(VK_NEXT)
    down = k.poll(down, 10.02)
    assert down and len(got) == 1
    down = k.poll(down, 10.04)                      # still held: not another press
    assert len(got) == 1
    keys.clear()
    down = k.poll(down, 10.06)
    keys.add(VK_NEXT)                               # contact bounce 40 ms later
    down = k.poll(down, 10.08)
    assert len(got) == 1
    keys.clear()
    down = k.poll(down, 10.5)
    keys.add(VK_PRIOR)                              # a real second press (the other clicker button)
    k.poll(down, 10.6)
    assert len(got) == 2


def test_a_key_held_at_startup_is_not_a_press():
    got = []
    k = GoKey(lambda: got.append(1), state_fn=lambda vk: True)
    k.poll(True, 1.0)
    assert not got
