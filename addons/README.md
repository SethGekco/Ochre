# Addons

Drop an addon directory here. Each needs an `addon.ini` and an entry module
exposing `register(host)`.

Nothing in this directory runs until Ochre is asked to load it, and a
never-before-seen addon is not loaded unattended — see `data/addons.ini`.

    addons/
      my-addon/
        addon.ini
        __init__.py        def register(host): ...
        data/*.ini         the addon's own data, read via host.data_path()

The full contract is `ochre/addons/api.py`; it is small on purpose.
