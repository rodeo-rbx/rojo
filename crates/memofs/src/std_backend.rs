use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::{
    collections::{HashMap, HashSet},
    io,
};

use crossbeam_channel::Receiver;
use notify::{watcher, DebouncedEvent, RecommendedWatcher, RecursiveMode, Watcher};

use crate::{DirEntry, Metadata, ReadDir, VfsBackend, VfsEvent};

/// `VfsBackend` that uses `std::fs` and the `notify` crate.
pub struct StdBackend {
    watcher: RecommendedWatcher,
    watcher_receiver: Receiver<VfsEvent>,
    /// Each recursively watched root, mapped to its canonical path.
    watches: HashMap<PathBuf, PathBuf>,
    /// The canonical paths of the watched roots.
    canonical_watches: HashSet<PathBuf>,
}

impl StdBackend {
    pub fn new() -> io::Result<StdBackend> {
        let (notify_tx, notify_rx) = mpsc::channel();
        let watcher = watcher(notify_tx, Duration::from_millis(50)).map_err(io::Error::other)?;

        let (tx, rx) = crossbeam_channel::unbounded();

        thread::spawn(move || {
            for event in notify_rx {
                match event {
                    DebouncedEvent::Create(path) => {
                        tx.send(VfsEvent::Create(path))?;
                    }
                    DebouncedEvent::Write(path) => {
                        tx.send(VfsEvent::Write(path))?;
                    }
                    DebouncedEvent::Remove(path) => {
                        tx.send(VfsEvent::Remove(path))?;
                    }
                    DebouncedEvent::Rename(from, to) => {
                        tx.send(VfsEvent::Remove(from))?;
                        tx.send(VfsEvent::Create(to))?;
                    }
                    _ => {}
                }
            }

            Result::<(), crossbeam_channel::SendError<VfsEvent>>::Ok(())
        });

        Ok(Self {
            watcher,
            watcher_receiver: rx,
            watches: HashMap::new(),
            canonical_watches: HashSet::new(),
        })
    }
}

impl StdBackend {
    /// Watches `path` recursively. It's recorded before the watcher is asked,
    /// so a path that fails to watch isn't tried again on every read.
    fn watch_root(&mut self, path: &Path) -> io::Result<()> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.canonical_watches.insert(canonical.clone());
        self.watches.insert(path.to_path_buf(), canonical);
        self.watcher
            .watch(path, RecursiveMode::Recursive)
            .map_err(io::Error::other)
    }
}

impl VfsBackend for StdBackend {
    fn read(&mut self, path: &Path) -> io::Result<Vec<u8>> {
        fs_err::read(path)
    }

    fn write(&mut self, path: &Path, data: &[u8]) -> io::Result<()> {
        fs_err::write(path, data)
    }

    fn exists(&mut self, path: &Path) -> io::Result<bool> {
        std::fs::exists(path)
    }

    fn read_dir(&mut self, path: &Path) -> io::Result<ReadDir> {
        let entries: Result<Vec<_>, _> = fs_err::read_dir(path)?.collect();
        let mut entries = entries?;

        entries.sort_by_cached_key(|entry| entry.file_name());

        let inner = entries
            .into_iter()
            .map(|entry| Ok(DirEntry { path: entry.path() }));

        Ok(ReadDir {
            inner: Box::new(inner),
        })
    }

    fn create_dir(&mut self, path: &Path) -> io::Result<()> {
        fs_err::create_dir(path)
    }

    fn create_dir_all(&mut self, path: &Path) -> io::Result<()> {
        fs_err::create_dir_all(path)
    }

    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        fs_err::remove_file(path)
    }

    fn remove_dir_all(&mut self, path: &Path) -> io::Result<()> {
        fs_err::remove_dir_all(path)
    }

    fn metadata(&mut self, path: &Path) -> io::Result<Metadata> {
        let inner = fs_err::metadata(path)?;

        Ok(Metadata {
            is_file: inner.is_file(),
        })
    }

    fn canonicalize(&mut self, path: &Path) -> io::Result<PathBuf> {
        fs_err::canonicalize(path)
    }

    fn event_receiver(&self) -> crossbeam_channel::Receiver<VfsEvent> {
        self.watcher_receiver.clone()
    }

    fn watch(&mut self, path: &Path) -> io::Result<()> {
        if !path
            .ancestors()
            .any(|ancestor| self.watches.contains_key(ancestor))
        {
            return self.watch_root(path);
        }

        // A recursive watch stops at symlinks (FSEvents and inotify don't
        // follow them), so a path reached through one out of every watched
        // tree, like a pesde workspace package linked into roblox_packages,
        // is watched at its target too. Its events then arrive at the
        // target's path, which is the path snapshots record: their relevant
        // paths are canonical.
        let canonical = match std::fs::canonicalize(path) {
            Ok(canonical) => canonical,
            Err(_) => return Ok(()),
        };
        if canonical
            .ancestors()
            .any(|ancestor| self.canonical_watches.contains(ancestor))
        {
            return Ok(());
        }
        self.watch_root(&canonical)
    }

    fn unwatch(&mut self, path: &Path) -> io::Result<()> {
        if let Some(canonical) = self.watches.remove(path) {
            self.canonical_watches.remove(&canonical);
        }
        self.watcher.unwatch(path).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    /// A directory reached through a symlink out of the watched tree reports
    /// its changes at the target's path, which is where snapshots record them.
    #[cfg(unix)]
    #[test]
    fn watches_through_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        let target = dir.path().join("target");
        fs_err::create_dir(&root).unwrap();
        fs_err::create_dir(&target).unwrap();
        fs_err::write(target.join("file.luau"), "a").unwrap();
        std::os::unix::fs::symlink(&target, root.join("link")).unwrap();

        let mut backend = StdBackend::new().unwrap();
        let receiver = backend.event_receiver();
        backend.watch(&root).unwrap();
        backend.watch(&root.join("link")).unwrap();
        thread::sleep(Duration::from_millis(250));

        fs_err::write(target.join("file.luau"), "b").unwrap();

        let expected = fs_err::canonicalize(target.join("file.luau")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(VfsEvent::Write(path)) | Ok(VfsEvent::Create(path)) if path == expected => break,
                Ok(_) => continue,
                Err(_) => panic!("no event for {}", expected.display()),
            }
        }
    }
}
